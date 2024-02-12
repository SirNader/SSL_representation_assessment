from .base.freezer_base import FreezerBase


class DinoHeadLastLayerFreezer(FreezerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return type(self).__name__

    def _change_state(self, model, requires_grad):
        for p in model.last_layer.parameters():
            p.requires_grad = requires_grad
