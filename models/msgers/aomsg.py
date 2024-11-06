# decoder style AoMSG
# set the module as a separate file for cleaness
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from models.loss import get_match_idx, get_association_sv
import numpy as np
from models.loss import InfoNCELoss, MaskBCELoss, FocalLoss, MaskMetricLoss, MeanSimilarityLoss, TotalCodingRate

class DecoderAssociator(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_heads, num_layers, object_dim, place_dim, 
                 num_img_patches, model, pr_loss, obj_loss, **kwargs):
        super(DecoderAssociator, self).__init__()
        self.model_name = model
        self.object_dim = object_dim
        self.place_dim = place_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_img_patches = num_img_patches # 256 # 224//14 ** 2 [CLS]

        # self.sep_token = nn.Parameter(torch.empty(1, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = hidden_dim,
            nhead = num_heads,
            dim_feedforward = int(hidden_dim * 4),
            dropout = 0.1,
            activation = 'gelu',
            layer_norm_eps = 1e-5,
            batch_first=True, 
            norm_first=False,
        )
        decoder_norm = nn.LayerNorm(hidden_dim, eps=1e-5, elementwise_affine=True) # 1e-5 or 1e-6?
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers, norm=decoder_norm)

        # box embedding
        self.box_emb = nn.Linear(4, hidden_dim, bias=False)
        self.whole_box = nn.Parameter(torch.tensor([0, 0, 224, 224], dtype=torch.float32), requires_grad=False)

        # input adaptor
        self.object_proj = nn.Linear(object_dim, hidden_dim, bias=False)
        self.place_proj = nn.Linear(place_dim, hidden_dim, bias=False)
        # output head
        # self.object_head = nn.Sequential(
        #     nn.Linear(hidden_dim, output_dim),
        #     nn.GELU(approximate='tanh'),
        #     nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-5),
        #     nn.Linear(output_dim, output_dim),
        # )
        self.object_head = nn.Linear(hidden_dim, output_dim)
        
        # self.place_head = nn.Sequential(
        #     nn.Linear(hidden_dim, output_dim),
        #     nn.GELU(approximate='tanh'),
        #     nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-5),
        #     nn.Linear(output_dim, output_dim),
        # )
        self.place_head = nn.Linear(hidden_dim, output_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_img_patches + 1, hidden_dim), requires_grad=False)

        self.measure_cos_pp = False
        if pr_loss == "bce":
            w = kwargs["pp_weight"] if "pp_weight" in kwargs else 1.0
            self.pr_loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([w]))
            self.measure_cos_pp = False
        else:
            self.pr_loss_fn = nn.MSELoss(reduction='none')
            self.measure_cos_pp = True

        self.measure_cos_obj = False
        if obj_loss == "bce":
            assert "pos_weight" in kwargs
            pos_weight = kwargs["pos_weight"]
            self.obj_loss_fn = MaskBCELoss(pos_weight=pos_weight)
            self.measure_cos_obj = False
        elif obj_loss == "focal":
            assert "alpha" in kwargs
            assert "gamma" in kwargs
            self.obj_loss_fn = FocalLoss(alpha=kwargs["alpha"], gamma=kwargs["gamma"])
            self.measure_cos_obj = False
        elif obj_loss == "infonce":
            assert "temperature" in kwargs
            self.obj_loss_fn = InfoNCELoss(temperature=kwargs["temperature"], learnable=False)
            self.measure_cos_obj = False
        else:
            self.obj_loss_fn = MaskMetricLoss()
            self.measure_cos_obj = True
        self.obj_loss_fn_sim = MeanSimilarityLoss()
        self.obj_tcr = TotalCodingRate(eps=0.2)

        self.initialize_weights()


    def initialize_weights(self,):
        
        self.apply(self._init_weights)

        grid_size = int(self.num_img_patches**.5)
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1], 
            grid_size=grid_size, 
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    # reference: from MAE's code base 
    # https://github.com/facebookresearch/mae/blob/main/models_mae.py#L68
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def pad_objects(self, object_emb):
        padded_object_emb = pad_sequence(object_emb, batch_first=True, padding_value=0)
        return padded_object_emb
    
    def get_query_mask(self, padded_object_emb):
        """
        Obtain masking for attention, since object embedding are padded.
        padded_object_emb: the real 0/1 masking
        """
        B, L, Ho = padded_object_emb.size()
        img_length = 1 # just 1 token for the whole-box query
        obj_mask = (padded_object_emb == 0).all(dim=-1).to(padded_object_emb.device)
        place_mask = torch.zeros(B, img_length, dtype = obj_mask.dtype, device = obj_mask.device)
        total_mask = torch.cat((place_mask, obj_mask), dim=1)
        return total_mask


    def forward(self, object_emb, place_emb, detections):
        """
        input:
            object_emb: list of B elements, each is a tensor of embeddings (K x Ho) of that image but in various lengths K.
            detectoins: list of B elements, each is a tensor of detections (K x 4) of that image, in various lengths K.
            place_emb: B x L x Hp, or B x H x W x Hp, D is the dimension of the place embeddings
        output:
            object_association_loss, place_recognition_loss
        """
        # pad object
   

        padded_obj_embd = self.pad_objects(object_emb)
        # print(padded_obj_embd.shape)
        B, K, Ho = padded_obj_embd.shape
        padded_obj_box = self.pad_objects(detections)
        
        whole_box_expanded = self.whole_box.unsqueeze(0).expand(B, 1, -1)
        
        query = torch.cat([whole_box_expanded, padded_obj_box], dim = 1) / 224.0 # hard-code nomalization
        # convert to embedding
        query = self.box_emb(query)

        query_mask = self.get_query_mask(padded_obj_embd) # B x K + 1 -> 1 for the whole_box preppended



        # flatten place
        if len(place_emb.size()) == 4:
            Hp = place_emb.size(1)
            place_emb = torch.einsum("bchw -> bhwc", place_emb)
            place_emb = place_emb.view(B, -1, Hp)

        # object and place embeddings, adapt to dimension
        object_feat = self.object_proj(padded_obj_embd) # B x K x D

        # place embeddings
        place_feat = self.place_proj(place_emb) # B x M x D
        
        # condition the query with embedding
        conditioning = torch.cat([place_feat.mean(dim=1, keepdim=True), object_feat], dim=1)
        query = query + conditioning

        memory = place_feat + self.pos_embed[:, :place_feat.size(1), :]

        # decoding
        decoded_emb = self.decoder(
            tgt = query,
            memory = memory,
            tgt_key_padding_mask = query_mask,
        )

        #object and place predictions
        place_enc = self.place_head(decoded_emb[:, 0, :]) # B x h

        object_enc = self.object_head(decoded_emb[:, 1:, :]) # B x K x h

        place_logits = self.predict_place(place_enc)
        object_logits = self.predict_object(object_enc)

        results = {
            'embeddings': object_enc,
            'place_embeddings': place_enc,
            'place_predictions': place_logits,
            'object_predictions': object_logits,
        }
        
        return results


    def predict_object(self, padded_obj_feat):
        # object_embeddings: B x K x Ho, padded from object embedding
        B, K, H = padded_obj_feat.size()
        # # object_predictions: BK x BK
        if self.measure_cos_obj:
            norm = padded_obj_feat.norm(dim=-1, keepdim=True)
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            normed_obj_feat = padded_obj_feat / norm # B x K x Ho
        # just dot product:
        else:
            normed_obj_feat = padded_obj_feat
        # --- #
        flatten_obj_feat = normed_obj_feat.view(-1, H) # flatten the first two dimension
        object_predictions = flatten_obj_feat @ flatten_obj_feat.t() # BK x BK
        return object_predictions  
    
    def predict_place(self, place):
        # simple cosine similarity

        # place_predictions: B x B
        if self.measure_cos_pp:
            normed_p = place / place.norm(dim=-1, keepdim=True)
        else:
            normed_p = place
        place_logits = normed_p @ normed_p.t()
        return place_logits
    
    def get_loss(self, results, additional_info, match_inds, place_labels, weights):
        # prepare
        num_emb = results['embeddings'].size(1)
        reorderd_idx = get_match_idx(match_inds, additional_info, num_emb)
        logs = {}
        # get loss
        # object similarity loss with TCR regularizer
        sim_loss, mean_dis, tcr, id_counts = self.object_similarity_loss(results['embeddings'], reorderd_idx)
        logs['tcr'] = tcr.item()
        logs['obj_sim_loss'] = sim_loss.item()
        # logs['num_obj'] = id_counts.shape[0]
        logs['mean_dis'] = mean_dis.item()
        # logs['avg_num_instances'] = id_counts.sum().item() / (id_counts.shape[0] + 1e-5)

        # object_loss = weights['obj'] * sim_loss + weights['mean'] * mean_dis + weights['tcr'] * tcr
        # # object association loss
        object_loss = self.object_association_loss(results['object_predictions'], reorderd_idx)

        logs['running_loss_obj'] = object_loss.item()
        
        # place recognition loss
        place_loss = self.place_recognition_loss(results['place_predictions'], place_labels)

        total_loss = object_loss + weights['pr'] * place_loss #  weights['tcr'] * tcr
        logs['running_loss_pr'] = place_loss.item()
        # print(sim_loss, mean_dis, tcr, object_loss, place_loss)
        return total_loss, logs

    def place_recognition_loss(self, place_predictions, place_labels): # TODO: check implementation
        # place_predictions: B x 1
        # place_labels: B x 1
        # loss: scalar
        # place_predictions = (place_predictions + 1.0) / 2.0
        
        loss = self.pr_loss_fn(place_predictions, place_labels).mean()
        return loss
    
    def object_association_loss(self, object_predictions, reorderd_idx):
        """
        input:
            object_predictions: BN x BN, cosine similarity matrix
            reorder_idx: BN, padded, reordered gt_indices to match the pred_indices
        intermediate:
            supervision_matrix: BN x BN, binary matrix indicating the object association
            mask: BN x BN, binary matrix indicating the valid entries in the supervision_matrix
        output:
            loss: scalar
        """
        # supervision is already masked by the mask
        supervision_matrix, mask = get_association_sv(reorderd_idx)

        # using wrapped loss
        total_loss = self.obj_loss_fn(object_predictions, supervision_matrix, mask)
        return total_loss
    
    def object_similarity_loss(self, embeddings, matched_idx):
        """
        compute the similarity loss and the regularization
        """
        B, N, h = embeddings.size()
        flatten_embeddings = embeddings.view(-1, h)
        sim_loss, mean_dis_loss, id_counts = self.obj_loss_fn_sim(flatten_embeddings, matched_idx)
        tcr = self.obj_tcr(flatten_embeddings, matched_idx)
        return sim_loss, mean_dis_loss, tcr, id_counts
    

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
