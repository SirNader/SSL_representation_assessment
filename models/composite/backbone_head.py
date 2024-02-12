import torch.nn as nn

from models import model_from_kwargs
from models.poolings import pooling_from_kwargs
from utils.factory import create
from utils.model_utils import get_output_shape_of_model
from ..base.composite_model_base import CompositeModelBase


class BackboneHead(CompositeModelBase):
    def __init__(self, backbone, head, pooling=None, **kwargs):
        super().__init__(**kwargs)
        self.backbone = create(
            backbone,
            model_from_kwargs,
            stage_path_provider=self.stage_path_provider,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
        )
        self.pooling = create(pooling, pooling_from_kwargs, model=self.backbone) or nn.Identity()
        forward_fn = lambda x: self.pooling(self.backbone.features(x))
        self.latent_shape = get_output_shape_of_model(model=self.backbone, forward_fn=forward_fn)
        self.head = create(
            head,
            model_from_kwargs,
            stage_path_provider=self.stage_path_provider,
            input_shape=self.latent_shape,
            output_shape=self.output_shape,
            update_counter=self.update_counter,
        )

    @property
    def submodels(self):
        return dict(backbone=self.backbone, head=self.head)

    def forward(self, x, backbone_forward_kwargs=None):
        features = self.backbone.features(x, **(backbone_forward_kwargs or {}))
        pooled = self.pooling(features)
        return self.head(pooled)

    def features(self, x):
        features = self.backbone.features(x)
        pooled = self.pooling(features)
        return pooled

    def predict(self, x):
        features = self.features(x)
        return self.head.predict(features)

    def predict_binary(self, x, head_kwargs=None):
        features = self.features(x)
        return self.head.predict_binary(features, **(head_kwargs or {}))
