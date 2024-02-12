import numpy as np
import torch
import torch.nn as nn

from models.base.single_model_base import SingleModelBase
from models.poolings.single_pooling import SinglePooling
from utils.factory import create


class LinearHead(SingleModelBase):
    def __init__(self, nonaffine_batchnorm=False, pooling=None, **kwargs):
        super().__init__(**kwargs)
        self.nonaffine_batchnorm = nonaffine_batchnorm
        self.pooling = create(pooling, SinglePooling) or nn.Identity()
        input_shape = self.pooling(torch.ones(1, *self.input_shape)).shape[1:]
        input_dim = np.prod(input_shape)
        self.norm = nn.BatchNorm1d(input_dim, affine=False) if nonaffine_batchnorm else nn.Identity()
        self.layer = nn.Sequential(
            self.pooling,
            nn.Flatten(start_dim=1),
            self.norm,
            nn.Linear(input_dim, np.prod(self.output_shape)),
        )

    def forward(self, x):
        return self.layer(x)

    def features(self, x):
        return self(x)

    def predict(self, x):
        return dict(main=self(x))
