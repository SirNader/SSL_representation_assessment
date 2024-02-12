import torch
import torch.nn.functional as F

from distributed.config import get_rank
from distributed.gather import all_gather_grad


def simclr_loss_fn(projected, indexes, temperature):
    projected = F.normalize(projected, dim=-1)
    gathered_projected = all_gather_grad(projected)

    sim = torch.exp(torch.einsum("if,jf->ij", projected, gathered_projected) / temperature)

    gathered_indexes = all_gather_grad(indexes)

    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)
    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, projected.size(0) * get_rank():].fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim * neg_mask, 1)
    loss = -(torch.mean(torch.log(pos / (pos + neg))))
    return loss
