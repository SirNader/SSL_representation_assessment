from models.poolings import pooling2d_from_kwargs
from utils.vit_util import get_sequence_lengths, flatten_2d_to_1d, sequence_to_2d_with_seqlens
from .vit_mae import VitMae


class PooledVit(VitMae):
    def __init__(self, pool_before_block_indices, pooling, **kwargs):
        super().__init__(**kwargs)
        self.pool_before_block_indices = list(sorted(set(pool_before_block_indices)))
        assert len(self.pool_before_block_indices) == len(pool_before_block_indices)
        self.pooling = pooling2d_from_kwargs(kind=pooling, factor=2)
        _, h, w = self.input_shape
        self.h_seqlen, self.w_seqlen = get_sequence_lengths(patch_size=self.patch_size, img_height=h, img_width=w)

    def _before_block(self, x, idx):
        if idx in self.pool_before_block_indices:
            seqlen_denom = 2 ** self.pool_before_block_indices.index(idx)
            patch_tokens_as_img, aux_tokens = sequence_to_2d_with_seqlens(
                tokens=x,
                h_seqlen=int(max(1, self.h_seqlen / seqlen_denom)),
                w_seqlen=int(max(1, self.w_seqlen / seqlen_denom)),
                n_aux_tokens=1,
            )
            patch_tokens_as_img = self.pooling(patch_tokens_as_img)
            x = flatten_2d_to_1d(patch_tokens_as_img=patch_tokens_as_img, aux_tokens=aux_tokens)
        return x
