import einops
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy


class PatchwiseRotationLoss(nn.Module):
    @staticmethod
    def forward(prediction, target):
        assert prediction.ndim == 3
        # cross entropy expects [batch_size, n_classes, n_patches]
        prediction = einops.rearrange(prediction, "bs n_patches n_classes -> bs n_classes n_patches")
        loss = F.cross_entropy(prediction, target)
        acc = accuracy(prediction, target)
        return loss, dict(accuracy=acc)
