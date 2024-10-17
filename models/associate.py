# object association
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from models.loss import get_match_idx, get_association_sv
from models.loss import InfoNCELoss, MaskBCELoss, MaskMetricLoss, MeanSimilarityLoss, TotalCodingRate
import numpy as np
from models.msgers.aomsg import DecoderAssociator

# from timm.layers import Mlp


class SepAssociator(nn.Module):
    """
    takes place and object features for association training
    0) separate object association and place recognition
    1) a self-attention model and two task head
    """
    def __init__(self, object_model, place_model, object_dim, place_dim, output_dim, model="SepMSG",
                 pr_loss='mse', obj_loss='bce', **kwargs):
        super(SepAssociator, self).__init__()
        self.model_name = model
        self.object_model = object_model
        self.place_model = place_model
        self.object_dim = object_dim
        self.place_dim = place_dim
        self.output_dim = output_dim
        
        self.measure_cos_pp = False
        if pr_loss == "bce":
            w = kwargs["pp_weight"] if "pp_weight" in kwargs else 1.0
            self.pr_loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([w]))
            self.measure_cos_pp = False
        else:
            self.pr_loss_fn = nn.MSELoss(reduction='none')
            self.measure_cos_pp = True


        self.obj_loss_fn_sim = MeanSimilarityLoss()
        self.obj_tcr = TotalCodingRate(eps=0.2)

        self.measure_cos_obj = False
        if obj_loss == "bce":
            pos_weight = kwargs["pos_weight"]
            self.obj_loss_fn = MaskBCELoss(pos_weight=pos_weight)
            self.measure_cos_obj = False
        elif obj_loss == "infonce":
            self.obj_loss_fn = InfoNCELoss(temperature=kwargs["temperature"], learnable=False)
            self.measure_cos_obj = False
        else:
            self.obj_loss_fn = MaskMetricLoss()
            self.measure_cos_obj = True
            

        if object_model == "identity":
            self.object_model = nn.Identity()
        elif object_model == "linear":
            self.object_model = nn.Linear(object_dim, output_dim)
        elif object_model == "mlp":
            self.object_model = nn.Sequential(
                nn.Linear(object_dim, output_dim),
                nn.GELU(approximate='tanh'),
                nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-5),
                nn.Linear(output_dim, output_dim),
            )
            # self.object_model = nn.Sequential(
            #     # mlp head from ViT
            #     Mlp(in_features=object_dim, hidden_features=int(object_dim*4), act_layer=nn.GELU, drop=0.),
            #     nn.LayerNorm(object_dim, elementwise_affine=False, eps=1e-5),
            #     nn.Linear(object_dim, output_dim),
            # )
        if place_model == "identity":
            self.place_model = nn.Identity()
        elif place_model == "linear":
            self.place_model = nn.Linear(place_dim, output_dim)
        elif place_model == "mlp":
            self.place_model = nn.Sequential(
                nn.Linear(place_dim, output_dim),
                nn.GELU(approximate='tanh'),
                nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-5),
                nn.Linear(output_dim, output_dim),
            )
            # self.place_model = nn.Sequential(
            #     # mlp head from ViT
            #     Mlp(in_features=place_dim, hidden_features=int(place_dim*4), act_layer=nn.GELU, drop=0.),
            #     nn.LayerNorm(place_dim, elementwise_affine=False, eps=1e-5),
            #     nn.Linear(place_dim, output_dim),
            # )

        self.initialize_weights()

    def initialize_weights(self,):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def pad_objects(self, object_emb):
        padded_object_emb = pad_sequence(object_emb, batch_first=True, padding_value=0)
        return padded_object_emb

    def predict_place(self, place_embeddings):
        # Option 1: simple cosine similarity
        # place_embeddings: B x Hp
        # place_predictions: B x 1
        if self.measure_cos_pp:
            normed_p = place_embeddings / place_embeddings.norm(dim=-1, keepdim=True) # B x Hp
        else:
            normed_p = place_embeddings
        place_logits = normed_p @ normed_p.t()
        return place_logits
    
    def predict_object(self, padded_obj_feat):
        # object_embeddings: B x K x Ho, padded from object embedding
        B, K, H = padded_obj_feat.size()
        # object_predictions: BK x BK
        if self.measure_cos_obj:
            norm = padded_obj_feat.norm(dim=-1, keepdim=True)
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            normed_obj_feat = padded_obj_feat / norm # B x K x Ho
        else:
            normed_obj_feat = padded_obj_feat
        flatten_obj_feat = normed_obj_feat.view(-1, H) # flatten the first two dimension
        # object_predictions = torch.matmul(padded_obj_feat, padded_obj_feat2.transpose(-1, -2))
        object_predictions = flatten_obj_feat @ flatten_obj_feat.t() # BK x BK
        return object_predictions
    
   

    def forward(self, object_emb, place_emb, boxes=None):
        """
        input:
            object_emb: list of B elements, each is a tensor of embeddings (K x Ho) of that image but in various lengths K.
            place_emb1, place_emb2: B x Hp, D is the dimension of the place embeddings
            supervision_matrix: B x N x M, binary matrix indicating the object association
            mask: B x N x M, binary matrix indicating the valid entries in the supervision_matrix
        output:
            results: a dictionary containing the embeddings and predictions
        """

        padded_object_emb = self.pad_objects(object_emb) # turn list of tensors KxH to tensor BxKxH
        # print("padded obj emb", padded_object_emb.size())
        object_feat = self.object_model(padded_object_emb)
        # print("object feature", object_feat.size())
        place_feat = self.place_model(place_emb)
        
        # prediction, across the batch
        object_logits = self.predict_object(object_feat)
        place_logits = self.predict_place(place_feat)

        results = {
            'embeddings': object_feat,
            'place_embeddings': place_feat,
            'place_predictions': place_logits,
            'object_predictions': object_logits,
        }
        
        return results
    
    def get_loss(self, results, additional_info, match_inds, place_labels, weights):
        """
        results: dict
        place_labels" int, B x B, binary
        weights: loss weights
        """
        # prepare
        num_emb = results['embeddings'].size(1)
        reorderd_idx = get_match_idx(match_inds, additional_info, num_emb)
        # print("total reordered index", reorderd_idx)
        logs = {}
        # association_sv, association_mask = get_association_sv(reorderd_idx)
        # get loss
        # torch.save(results['embeddings'].cpu(), "embeddings.pt")
        # object similarity loss with TCR regularizer
        sim_loss, mean_dis, tcr, id_counts = self.object_similarity_loss(results['embeddings'], reorderd_idx)

        logs['tcr'] = tcr.item()
        logs['obj_sim_loss'] = sim_loss.item()
        # logs['num_obj'] = id_counts.shape[0]
        logs['mean_dis'] = mean_dis.item()
        # logs['avg_num_instances'] = id_counts.sum().item() / (id_counts.shape[0] + 1e-5)

        # object_loss = weights['obj'] * sim_loss + weights['mean'] * mean_dis + weights['tcr'] * tcr
        
        # object association loss
        object_loss = self.object_association_loss(results['object_predictions'], reorderd_idx)

        logs['running_loss_obj'] = object_loss.item()
        
        # place recognition loss
        place_loss = self.place_recognition_loss(results['place_predictions'], place_labels)

        total_loss = object_loss + weights['pr'] * place_loss
        logs['running_loss_pr'] = place_loss.item()
        # print(sim_loss, mean_dis, tcr, object_loss, place_loss)
        return total_loss, logs
    

    def place_recognition_loss(self, place_predictions, place_labels): 
        # place_predictions: B x 1
        # place_labels: B x 1
        # loss: scalar
        
        place_predictions = (place_predictions + 1.0) / 2.0
        loss = self.pr_loss_fn(place_predictions, place_labels).mean()
        return loss
    
    def object_association_loss(self, object_predictions, reordered_idx):
        """
        input:
            object_predictions: BN x BN, cosine similarity matrix
            reordered_idx: BN, padded reordered object uid 
        intermediate:
            supervision_matrix: BN x BN, binary matrix indicating the object association
            mask: BN x BN, binary matrix indicating the valid entries in the supervision_matrix
        output:
            loss: scalar
        """

        # supervision is already masked by the mask
        supervision_matrix, mask = get_association_sv(reordered_idx)

        # valid_object_predictions = object_predictions * mask
        # loss = self.obj_loss_fn(valid_object_predictions, supervision_matrix)
        # masked_loss = loss * mask
        # total_loss = masked_loss.sum() / (mask.sum() + 1e-9) # mask could be all 0 if there is no object!

        # using wrapped loss
        total_loss = self.obj_loss_fn(object_predictions, supervision_matrix, mask)
        return total_loss
    
    def object_similarity_loss(self, embeddings, matched_idx):
        """
        compute the similarity loss and the regularization
        """
        B, N, h = embeddings.size()
        # print(B, N, h)
        flatten_embeddings = embeddings.view(-1, h)
        # assert flatten_embeddings.size(0) == matched_idx.size(0)
        sim_loss, mean_dis_loss, id_counts = self.obj_loss_fn_sim(flatten_embeddings, matched_idx)
        tcr = self.obj_tcr(flatten_embeddings, matched_idx)
        return sim_loss, mean_dis_loss, tcr, id_counts
   


# ----------------- helper functions -----------------

def get_1d_sincos_pos_embed(embed_dim, length):
    """
    embed_dim: output dimension for each position
    length: max sequence length (including special tokens) that are used to convert to positions, M
    -> pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    pos = np.arange(length, dtype=np.float32)

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb




def build_encoder(args):
    hidden_dim = args['hidden_dim']
    num_heads = args['num_heads']
    ffn_dim = int(args['mlp_ratio'] * hidden_dim)
    activation = args['activation'] # gelu
    encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, 
                                               nhead=num_heads,
                                               dim_feedforward=ffn_dim, 
                                               activation=activation, 
                                               batch_first=True, 
                                               norm_first=False,
                                               layer_norm_eps=1e-5,
                                               dropout= 0.1,
                                               )
                                            #    bias = True) # this is not in pytorch version 2.0.0
    norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args['num_layers'], norm=norm)
    return transformer_encoder
    

# ----------------- model factory -----------------

# this one has no parameter hence not applied for training
def SepMSG_direct(**kwargs):
    return SepAssociator(object_model='identity', place_model='identity', **kwargs)

def SepMSG_linear(**kwargs):
    return SepAssociator(object_model='linear', place_model='linear', **kwargs)

def SepMSG_mlp(**kwargs):
    return SepAssociator(object_model='mlp', place_model='mlp', **kwargs)

def AoMSG_S_2(**kwargs):
    return DecoderAssociator(hidden_dim=384, num_heads=6, num_layers=2, num_img_patches=256, **kwargs)

def AoMSG_S_1(**kwargs):
    return DecoderAssociator(hidden_dim=384, num_heads=6, num_layers=1, num_img_patches=256, **kwargs)

def AoMSG_S_4(**kwargs):
    return DecoderAssociator(hidden_dim=384, num_heads=6, num_layers=4, num_img_patches=256, **kwargs)

def AoMSG_B_4(**kwargs):
    return DecoderAssociator(hidden_dim=768, num_heads=12, num_layers=4, num_img_patches=256, **kwargs)

Asso_models = {
    'SepMSG-direct': SepMSG_direct, 'SepMSG-linear': SepMSG_linear, 'SepMSG-mlp': SepMSG_mlp, 
    'AoMSG-S-2': AoMSG_S_2, 'AoMSG-S-1': AoMSG_S_1, 'AoMSG-S-4': AoMSG_S_4,
    'AoMSG-B-4': AoMSG_B_4,
}
