import einops

from models.poolings.single_pooling import SinglePooling
from .vit_mae import VitMae


class SslMaskedEncoder(VitMae):
    # noinspection PyMethodOverriding
    def forward(self, x, mask_generator, single_mask=False):
        if mask_generator is None:
            return super().forward(x)

        # calculate how many tokens along height/width dimension
        # needs to be done dynamically to support varying image input sizes (e.g. multi-crop)
        _, _, img_h, img_w = x.shape
        token_h = int(img_h / self.patch_embed.patch_size[0])
        token_w = int(img_w / self.patch_embed.patch_size[1])
        x = self.embed_and_add_pos(x)

        # undo patch_embed flattening
        # (patch_embed is set to flatten in order to not need to unflatten in inference/without mask)
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=token_h, w=token_w)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = mask_generator.get_mask(x, single_mask=single_mask)

        x = self.append_cls_token(x)
        x = self.apply_transformer_blocks(x)

        return dict(x=x, mask=mask, ids_restore=ids_restore)

    # noinspection PyMethodOverriding
    def features(self, x, pool_kind=None, mask_generator=None, single_mask=False):
        if mask_generator is not None:
            output = self(x, mask_generator=mask_generator, single_mask=single_mask)
            return SinglePooling.get_pool_fn(kind=pool_kind, model=self)(output["x"])
        return super().features(x, pool_kind=pool_kind)
