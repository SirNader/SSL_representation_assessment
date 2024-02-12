from .base.freezer_base import FreezerBase


class VitBlockFreezer(FreezerBase):
    def __init__(self, block_idxs, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(block_idxs, list) and all(isinstance(block_idx, int) for block_idx in block_idxs)
        self.block_idxs = block_idxs

    def __str__(self):
        return f"{type(self).__name__}(block_idxs={self.block_idxs})"

    def _change_state(self, model, requires_grad):
        for block_idx in self.block_idxs:
            block = model.blocks[block_idx]
            for param in block.parameters():
                param.requires_grad = requires_grad
