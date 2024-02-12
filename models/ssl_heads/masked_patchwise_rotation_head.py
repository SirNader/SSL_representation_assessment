from functools import partial

import torch.nn as nn

from models.base.single_model_base import SingleModelBase
from models.poolings.single_pooling import SinglePooling
from initializers.functional import initialize_linear_bias_to_zero


class MaskedPatchwiseRotationHead(SingleModelBase):
    def __init__(self, backbone=None, **kwargs):
        super().__init__(**kwargs)
        assert len(self.input_shape) == 2
        input_dim = self.input_shape[1]
        self.projector = nn.Linear(input_dim, 4)
        self.token_pooling = SinglePooling(kind="all_patches", model=backbone)

    @property
    def _requires_initializer(self):
        return False

    def _model_specific_initialization(self):
        self.apply(initialize_linear_bias_to_zero)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        x = self.token_pooling(x)
        x = self.projector(x)
        return x
