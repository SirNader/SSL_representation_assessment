from functools import partial

import torch.nn as nn

from models.base.single_model_base import SingleModelBase
from models.modules.flatten_2d_tokens import Flatten2dTokens
from models.poolings import pooling2d_from_kwargs
from models.poolings.single_pooling import SinglePooling
from initializers.functional import initialize_linear_bias_to_zero


class PatchwiseRotationHead(SingleModelBase):
    def __init__(self, rotation_patch_size=None, backbone=None, token_pooling_kind=None, **kwargs):
        super().__init__(**kwargs)
        assert len(self.input_shape) == 2
        input_dim = self.input_shape[1]
        self.projector = nn.Linear(input_dim, 4)
        assert backbone.patch_size[0] == backbone.patch_size[1], "nonsquare not supported yet"
        assert rotation_patch_size % backbone.patch_size[0] == 0
        pooling_factor = int(rotation_patch_size / backbone.patch_size[0])
        if pooling_factor == 1:
            assert token_pooling_kind is None
        self.token_pooling = nn.Sequential(
            SinglePooling(kind="image", model=backbone),
            pooling2d_from_kwargs(kind=token_pooling_kind, factor=pooling_factor),
            Flatten2dTokens(),
        )

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
