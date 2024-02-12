import torch.nn as nn

from initializers.base.initializer_base import InitializerBase


class Mocov3ProbeInitializer(InitializerBase):
    @property
    def should_apply_model_specific_initialization(self):
        return True

    def init_weights(self, model, **_):
        model.apply(self.apply_fn)
        self.logger.info(f"initialized {type(model).__name__} with weight=trunc_normal(std=0.01) bias=0")

    @staticmethod
    def apply_fn(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
