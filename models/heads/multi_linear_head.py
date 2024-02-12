from itertools import product

import torch.nn as nn

from models.base.composite_model_base import CompositeModelBase
from .linear_head import LinearHead


class MultiLinearHead(CompositeModelBase):
    def __init__(self, poolings=None, optimizers=None, initializers=None, nonaffine_batchnorm=False, **kwargs):
        super().__init__(**kwargs)
        search_space = product(
            poolings.items() if poolings is not None else [(None, None)],
            optimizers.items() if optimizers is not None else [(None, None)],
            initializers.items() if initializers is not None else [(None, None)],
        )
        layers = {}
        for (pool_name, pooling), (optim_name, optimizer), (init_name, initializer) in search_space:
            ctor_kwargs = dict(
                pooling=pooling,
                optim_ctor=optimizer,
                initializer=initializer,
                nonaffine_batchnorm=nonaffine_batchnorm,
            )
            layer = LinearHead(
                **ctor_kwargs,
                input_shape=self.input_shape,
                output_shape=self.output_shape,
                update_counter=self.update_counter,
                stage_path_provider=self.stage_path_provider,
                ctor_kwargs=dict(**ctor_kwargs, kind="heads.linear"),
            )
            names = []
            if pooling is not None:
                names.append(pool_name)
            if optimizer is not None:
                names.append(optim_name)
            if initializers is not None:
                names.append(init_name)
            name = "_".join(names)

            layers[name] = layer
        self.layers = nn.ModuleDict(layers)

    @property
    def submodels(self):
        return self.layers

    def forward(self, x):
        return {key: layer(x) for key, layer in self.layers.items()}

    def features(self, x):
        return self(x)

    def predict(self, x):
        result = {}
        for key, layer in self.layers.items():
            result[key] = layer.predict(x)["main"]
        return result
