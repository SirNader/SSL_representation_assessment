import torch
from torch.nn.functional import interpolate

from utils.param_checking import to_2tuple
from .base.mask_generator import MaskGenerator


class GridMaskGenerator(MaskGenerator):
    def __init__(self, mask_size=None, **kwargs):
        super().__init__(**kwargs)
        assert self.mask_ratio in [0.25, 0.5, 0.75]
        if mask_size is not None:
            self.mask_size = to_2tuple(mask_size)
            assert isinstance(self.mask_size[0], int) and self.mask_size[0] > 0
            assert isinstance(self.mask_size[1], int) and self.mask_size[1] > 0
        else:
            self.mask_size = (1, 1)

    def __str__(self):
        mask_size_str = "" if self.mask_size != (1, 1) else f"mask_size=({self.mask_size[0]},{self.mask_size[1]})"
        return f"{type(self).__name__}({mask_size_str}{self._base_param_str})"

    def generate_noise(self, x, generator=None):
        N, _, H, W = x.shape
        # create mask of one block
        mask_h, mask_w = self.mask_size
        assert H % mask_h == 0 and H / mask_h >= 2
        assert W % mask_w == 0 and W / mask_w >= 2

        h_stride = 2
        w_stride = 2
        noise = torch.rand(N, h_stride, w_stride, device=x.device, generator=generator)
        if mask_h > 1 or mask_w > 1:
            h_stride *= mask_h
            w_stride *= mask_w
            noise = interpolate(noise.unsqueeze(1), scale_factor=(mask_h, mask_w), mode="nearest").squeeze(1)
        # repeat to full size
        repeat_h = int(H / h_stride)
        repeat_w = int(W / w_stride)
        noise = noise.repeat(1, repeat_h, repeat_w)
        return noise
