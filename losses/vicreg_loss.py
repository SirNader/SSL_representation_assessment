import torch
import torch.nn.functional as F

from distributed.gather import all_gather_grad


def vicreg_loss_fn(projected0, projected1, sim_loss_weight, var_loss_weight, cov_loss_weight):
    sim_loss = F.mse_loss(projected0, projected1)

    # vicreg's official code gathers the tensors here
    # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    gathered_projected0 = all_gather_grad(projected0)
    gathered_projected1 = all_gather_grad(projected1)

    # variance loss
    eps = 1e-4
    std_projected0 = (gathered_projected0.var(dim=0) + eps).sqrt()
    std_projected1 = (gathered_projected1.var(dim=0) + eps).sqrt()
    std_loss = F.relu(1 - std_projected0).mean() + F.relu(1 - std_projected1).mean()

    # covariance loss
    N, D = gathered_projected0.shape
    centered_gathered_projected0 = gathered_projected0 - gathered_projected0.mean(dim=0)
    centered_gathered_projected1 = gathered_projected1 - gathered_projected1.mean(dim=0)
    cov_projected0 = (centered_gathered_projected0.T @ centered_gathered_projected0) / (N - 1)
    cov_projected1 = (centered_gathered_projected1.T @ centered_gathered_projected1) / (N - 1)

    diag = torch.eye(D, device=projected0.device)
    cov_loss = cov_projected0[~diag.bool()].pow_(2).sum() / D + cov_projected1[~diag.bool()].pow_(2).sum() / D

    loss = sim_loss_weight * sim_loss + var_loss_weight * std_loss + cov_loss_weight * cov_loss
    return loss
