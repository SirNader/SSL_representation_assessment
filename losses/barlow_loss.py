import torch
import torch.nn.functional as F

from distributed.gather import all_reduce_mean_grad


def barlow_loss_fn(projected0, projected1, off_diagonal_scale):
    N, D = projected0.shape

    projected0 = F.batch_norm(projected0, running_mean=None, running_var=None, training=True)
    projected1 = F.batch_norm(projected1, running_mean=None, running_var=None, training=True)

    corr = torch.einsum("bi,bj->ij", projected0, projected1) / N

    all_reduce_mean_grad(corr)

    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    cdif[~diag.bool()] *= off_diagonal_scale
    # we don't scale the loss here as each contrastive head has its own weight for the loss
    loss = cdif.sum()
    return loss
