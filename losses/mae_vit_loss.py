import torch.nn as nn

from losses import loss_fn_from_kwargs
from utils.factory import create
from utils.vit_util import patchify_as_1d


class MaeVitLoss(nn.Module):
    def __init__(self, normalize_pixels, loss_function=None):
        super().__init__()
        self.normalize_pixels = normalize_pixels
        self.loss_function = create(loss_function, loss_fn_from_kwargs, reduction="none")
        if self.loss_function is None:
            self.loss_function = nn.MSELoss(reduction="none")
        assert self.loss_function.reduction == "none"

    def forward(self, prediction, target, mask, patch_size):
        # multi-view case
        if isinstance(prediction, list):
            return {f"view{i}": self(*args, patch_size) for i, args in enumerate(zip(prediction, target, mask))}

        patchified_target = patchify_as_1d(imgs=target, patch_size=patch_size)

        # normalize reconstructed pixels
        if self.normalize_pixels:
            mean = patchified_target.mean(dim=-1, keepdim=True)
            var = patchified_target.var(dim=-1, keepdim=True)
            patchified_target = (patchified_target - mean) / (var + 1.e-6) ** .5

        # unreduced loss
        loss = self.loss_function(prediction, patchified_target)
        # [batch_size, n_patches, c*prod(patch_size))] -> [batch_size, n_patches] (mean loss per patch)
        loss = loss.mean(dim=-1)
        # mean loss on removed patches (mask is 1 if the patch was removed)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


# TODO only needed for old contrastive impl
def mae_vit_loss(x, x_hat, patch_size, normalize_pixels, base_loss_function, mask):
    patchified_x = patchify_as_1d(imgs=x, patch_size=patch_size)

    # normalize reconstructed pixels
    if normalize_pixels:
        mean = patchified_x.mean(dim=-1, keepdim=True)
        var = patchified_x.var(dim=-1, keepdim=True)
        patchified_x = (patchified_x - mean) / (var + 1.e-6) ** .5
    # unreduced loss
    loss = base_loss_function(x_hat, patchified_x)
    # [batch_size, n_patches, c*prod(patch_size))] -> [batch_size, n_patches] (mean loss per patch)
    loss = loss.mean(dim=-1)
    # mean loss on removed patches (mask is 1 if the patch was removed)
    # loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)
    # TODO
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss
