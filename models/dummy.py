import numpy as np
import torch.nn as nn

from .base.single_model_base import SingleModelBase


class Dummy(SingleModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer = nn.Linear(np.prod(self.input_shape), np.prod(self.output_shape))

    def forward(self, x):
        return self.layer(x.flatten(start_dim=1))

    def predict(self, x):
        return dict(main=self(x))

    def predict_binary(self, x):
        return dict(main=self(x))
