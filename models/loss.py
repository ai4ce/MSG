# loss modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------  supervision generation ----------------------------- #
# reorder the object ids at gt_indices to pred_indices
def get_match_idx(match_indices,
                  info,
                    N):
    B = len(match_indices)
    # N = info['gt_bbox'].shape[1]
    # total_N = B * N
    total_reorderd_indx = []
    for bi in range(B):
        pred_indices = match_indices[bi][0]
        gt_indices = match_indices[bi][1]
        # reorder the object ids at gt_indices to pred_indices
        ori_obj_idx = info['obj_idx'][bi]
        # reordered_obj_ids = torch.full_like(ori_obj_idx, -1) # -1
        reordered_obj_ids = torch.full((N,), -1, dtype = ori_obj_idx.dtype, device = ori_obj_idx.device) # -1
        reordered_obj_ids[pred_indices] = ori_obj_idx[gt_indices]

        # print("-------- print each data --------", bi, "--------------")
        # print("pred index", pred_indices)
        # print("gt index", gt_indices)
        # print("original", ori_obj_idx)
        # print("mask", info['mask'][bi])
        # print("reordered idx", reordered_obj_ids)

        total_reorderd_indx.append(reordered_obj_ids)

    total_reorderd_indx = torch.cat(total_reorderd_indx, dim=0) # total_N = B*N = BN

    return total_reorderd_indx


def get_association_sv(total_reorderd_indx):
    """
    Based on the indicies from object bounding box matching, reorder the object_ids (instance ids), 
    and generate the association supervision for the object embeddings
    reorder and align ground truth object instances for association supervision
    input:
        match_indicies: a list of indices pairs obtained from hungarian matching, len(match_indices) == batch_size
        each indicies pair is (pred_indicies, gt_indicies), pred_indices == tensor([idx, idx, idx, ...]), gt_indices == tensor([idx, idx, idx, ...])
        info: dictionaries contaninig anotation information. 
                        info = {'gt_bbox': [batch_size, padded_num_objects, 4],
                                'obj_label': [batch_size, padded_num_objects],
                                'obj_idx': [batch_size, padded_num_objects],
                                'mask': [batch_size, padded_num_objects], # the mask for padding the objects to the same length
                                }
        N: the padded number of detected objects, obtained from object_embedding.size(1)
    output:
        batch_supervision_matrix: BN x BN binary object association supervision across the entire batch
        batch_mask: BN x BN binary masking applied over the supervision across the entire batch

    """

    expanded = total_reorderd_indx.unsqueeze(1) # (BN, 1)
    
    batch_equality = (expanded == expanded.transpose(0,1)).int() # (BN, BN)
    mask = (total_reorderd_indx == -1).int() * -1
    expanded_mask = mask.unsqueeze(1)
    batch_mask = (expanded_mask + expanded_mask.transpose(0,1) == 0).int()

    batch_supervision_matrix = (batch_equality * batch_mask).float()
    return batch_supervision_matrix, batch_mask

# -------------------------------- ******** --------------------------- #

# -------------------------------- loss modules ----------------------- #

class InfoNCELoss(nn.Module):
    """
    Computes the InfoNCE loss for object association (and maybe the place recognition too?)

    Hyper Parameters:
        - temperature: A scalar controlling the sharpness of the distribution.
    """
    def __init__(self, temperature=0.1, learnable=False):
        super(InfoNCELoss, self).__init__()
        # self.temp = temperature
        self.temp = nn.Parameter(torch.ones(1) * temperature)
        if learnable:
            self.temp.requires_grad = True
        else:
            self.temp.requires_grad = False

    def forward(self, predictions, supervision_matrix_masked, mask):
        """
        input:
            predictions: BN x BN, cosine similarity matrix
            supervision_matrix: BN x BN, binary matrix indicating the object association
            mask: BN x BN, binary matrix indicating the valid entries in the supervision_matrix
        output:
            loss: scalar
        """
        BN, _ = predictions.shape
        # print("size", BN)
        # Apply mask to object predictions and supervision matrix
        predictions_masked = predictions * mask
        # Calculate logits with temperature scaling
        logits = predictions_masked / self.temp
        # Calculate positive logits using the supervision matrix as a mask
        positive_mask = supervision_matrix_masked.to(dtype=torch.bool)
        positive_logits = logits[positive_mask] # (number of positives, )
        # print(logits.size(), positive_logits.size())
        # denominator (BN,) , all 0s are masked, and small value are added in case the previous masking put a whole row to 0
        logsumexp_denominator = torch.logsumexp(logits, dim=-1) + 1e-9
        # print("denom", logsumexp_denominator.size()) 
        # Calculate the loss for each positive logit
        losses = positive_logits - logsumexp_denominator[positive_mask.nonzero(as_tuple=True)[0]]
        
        # Average loss across all positives, considering only the valid entries
        valid_entries = mask.sum()
        loss = -torch.sum(losses) / valid_entries if valid_entries > 0 else torch.tensor(0.0).to(predictions.device)
        
        return loss

    

class MaskBCELoss(nn.Module):
    """
    BCE loss but have some logits masked due to invalidness, for example, embeddings for object unmatched
    """
    def __init__(self, pos_weight=5.0):
        super(MaskBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([pos_weight]))

    def forward(self, predictions, supervision, mask):
        valid_object_predictions = predictions # * mask
        # supervision is already masked by the mask
        # print(valid_object_predictions.type(), supervision_matrix.type())
        loss = self.bce_loss(valid_object_predictions, supervision)
        masked_loss = loss * mask
        total_loss = masked_loss.sum() / (mask.sum() + 1e-9) # mask could be all 0 if there is no object!
        return total_loss
    
class FocalLoss(nn.Module):
    """
    Focal loss. include masking
    Gamma is the focusing parameter that downweight the easy examples
    Alpha acts like the positive weighting in BCE loss, 
    while 1 - alpha acts as the weighting factor for the negative class
    """
    def __init__(self, alpha = 0.5, gamma = 2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, predictions, supervision, mask):
        """
        Args:
            prediction: (N, N), 
            target: (N, N), binary at each element
        """
        bceloss = self.bce_loss(predictions, supervision)
        pt = torch.exp(-bceloss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bceloss
        masked_loss = focal_loss * mask
        total_loss = masked_loss.sum() / (mask.sum() + 1e-9) 
        return total_loss
    
class MaskMetricLoss(nn.Module):
    """
    Metric loss but have some logits masked due to invalidness, for example, embeddings for object unmatched
    For now only simple MSE -- Euclidean
    Can be extended to other types of metric
    """
    def __init__(self,):
        super(MaskMetricLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, predictions, supervision, mask):
        # convert from [-1, 1] to [0, 1]
        valid_object_predictions = (predictions + 1.0) / 2 * mask
        # supervision is already masked by the mask
        # print(valid_object_predictions.type(), supervision_matrix.type())
        loss = self.loss_fn(valid_object_predictions, supervision)
        masked_loss = loss * mask
        total_loss = masked_loss.sum() / (mask.sum() + 1e-9) # mask could be all 0 if there is no object!
        return total_loss
    

class MeanSimilarityLoss(nn.Module):
    """
    This loss minimize the similarity between instances and their respective mean.
    """
    def __init__(self,):
        super(MeanSimilarityLoss, self).__init__()

    def mean_reg(self, means):
        # regularize the loss with mean distance in every batch
        # pull away the means
        n, h = means.shape
        # pairwise distance, scaled
        # mean_pair_dis = torch.cdist(means, means, p=2) / torch.sqrt(torch.tensor(h))
        # mean_dis = off_diagonal(mean_pair_dis).sum() / (n**2 + 1e-9)
        means_norm = torch.nn.functional.normalize(means, p=2, dim=1)
        cos_sim = torch.mm(means_norm, means_norm.t())
        mean_cos_sim = off_diagonal(cos_sim).sum() / (n**2 - n + 1e-9)
        return mean_cos_sim

    def forward(self, embeddings, flatten_idx):
        """
        Assume embeddings and idxes are all flattened already, BxN -> BN
        """
        # B, N, h = embeddings.size()
        # flatten_embeddings = embeddings.view(-1, h)
        # flatten_idx = obj_idx.flatten()

        h = embeddings.size(1)
        valid_entry = (flatten_idx != -1).float()

        unique_ids = flatten_idx.unique()
        unique_ids = unique_ids[unique_ids != -1]

        # obtain mean for each object (each unique id)
        embeddings_sum = torch.zeros((len(unique_ids), h), dtype=torch.float).to(embeddings.device)
        counts = torch.zeros(len(unique_ids), dtype=torch.float).to(embeddings.device)
        for i, unique_id in enumerate(unique_ids):
            id_mask = (flatten_idx == unique_id)
            embeddings_sum[i] = embeddings[id_mask].sum(dim=0)
            counts[i] = id_mask.sum()
        
        # assert embeddings_sum.requires_grad == True
        embeddings_mean = embeddings_sum / counts.unsqueeze(1)
        # restore shape, for distance computing
        embeddings_mean_expanded = torch.zeros_like(embeddings)
        for i, unique_id in enumerate(unique_ids):
            embeddings_mean_expanded[flatten_idx == unique_id] = embeddings_mean[i]

        cos_distance = 1 - F.cosine_similarity(embeddings, embeddings_mean_expanded)
        # print("cosine", cos_distance.size(), valid_entry.size())

        avg_dis = (cos_distance * valid_entry).sum() / (valid_entry.sum() + 1e-9)

        mean_dis = self.mean_reg(embeddings_mean) # minimize - distance => maximize distance


        return avg_dis, mean_dis, counts 


class TotalCodingRate(nn.Module):
    """
    A total coding rate regularizer
    """
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        # print("p,m", p, m)
        I = torch.eye(p,device=W.device)
        scalar = p*m / ((m+1e-5) * (m+1e-5) * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        # print("logdet", logdet)
        # assert not torch.isnan(logdet), torch.save(W.detach().cpu(), 'nan.pt')
        return logdet / 2.
    
    def forward(self, embeddings, idx):
        validity_mask = idx != -1
        # num_valids = validity_mask.sum()
        X = embeddings[validity_mask]

        # centralize each dimension
        # X = X - X.mean(dim=0)
        nX = F.normalize(X)
        return - self.compute_discrimn_loss(nX.T)



def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()