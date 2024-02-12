from .base.freezer_base import FreezerBase


class VitPatchEmbedFreezer(FreezerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return type(self).__name__

    def _change_state(self, model, requires_grad):
        model.patch_embed.proj.weight.requires_grad = requires_grad
        model.patch_embed.proj.bias.requires_grad = requires_grad
