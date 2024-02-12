import torch
import torch.nn as nn
import torch.nn.functional as F

from distributed.config import get_rank
from distributed.gather import all_gather_nograd
from utils.multi_crop_utils import multi_crop_loss


class Mocov3Loss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, projected, predicted):
        assert len(projected) == 2
        losses = multi_crop_loss(projected, predicted, self._forward)
        return {f"view{i}-view{j}": loss for (i, j), loss in losses.items()}

    def _forward(self, projected, predicted):
        batch_size = predicted.size(0)
        device = predicted.device

        predicted = F.normalize(predicted, dim=1)
        projected = F.normalize(projected, dim=1)

        # gather all targets without gradients
        projected = all_gather_nograd(projected)

        logits = torch.einsum("nc,mc->nm", [predicted, projected]) / self.temperature
        labels = torch.arange(batch_size, dtype=torch.long, device=device) + batch_size * get_rank()

        return F.cross_entropy(logits, labels) * 2 * self.temperature


# TODO only needed for old contrastive impl
def mocov3_loss_fn(predicted, projected, temperature):
    batch_size = predicted.size(0)
    device = predicted.device

    predicted = F.normalize(predicted, dim=1)
    projected = F.normalize(projected, dim=1)

    # gather all targets without gradients
    projected = all_gather_nograd(projected)

    logits = torch.einsum("nc,mc->nm", [predicted, projected]) / temperature
    labels = torch.arange(batch_size, dtype=torch.long, device=device) + batch_size * get_rank()

    return F.cross_entropy(logits, labels) * (2 * temperature)
