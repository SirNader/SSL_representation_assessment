import einops
import torch

from .base.mask_generator import MaskGenerator


class SingleBlockMaskGenerator(MaskGenerator):
    def generate_noise(self, x, generator=None):
        N, _, H, W = x.shape

        # calculate height/width
        if H == 1:
            # mask full height dimension
            block_h = 1
            block_w = W * self.mask_ratio
            assert block_w.is_integer()
            block_w = int(block_w)
        elif W == 1:
            # mask full width dimension
            block_w = 1
            block_h = H * self.mask_ratio
            assert block_h.is_integer()
            block_h = int(block_h)
        else:
            raise NotImplementedError
            # mask 75% of height/width (this does not correspond to 75% of the image)
            # block_h = H * self.mask_ratio
            # block_w = W * self.mask_ratio
            # assert block_h.is_integer()
            # assert block_w.is_integer()
            # block_h = int(block_h)
            # block_w = int(block_w)

        # sample idx
        i = einops.rearrange(torch.randint(0, H - block_h + 1, size=(N,), generator=generator), "n -> n 1 1")
        j = einops.rearrange(torch.randint(0, W - block_w + 1, size=(N,), generator=generator), "n -> n 1 1")

        noise = torch.rand(N, H, W, device=x.device, generator=generator)
        # high noise is masked
        h_idxs = einops.repeat(torch.arange(H), "h -> 1 h w", w=W)
        w_idxs = einops.repeat(torch.arange(W), "w -> 1 h w", h=H)
        h_idx_mask = torch.logical_and(i <= h_idxs, h_idxs < i + block_h)
        w_idx_mask = torch.logical_and(j <= w_idxs, w_idxs < j + block_w)
        inf = torch.full_like(noise, fill_value=float("inf"))
        noise = torch.where(torch.logical_and(h_idx_mask, w_idx_mask), inf, noise)

        return noise
