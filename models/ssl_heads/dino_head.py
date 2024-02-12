import torch
import torch.nn as nn

from models.modules.normalize import Normalize
from .base.momentum_projector_head import MomentumProjectorHead


class DinoHead(MomentumProjectorHead):
    def __init__(self, bottleneck_dim, **kwargs):
        # initialize variables for creating projector (done in ctor of baseclass)
        self.last_layer = None
        self.bottleneck_dim = bottleneck_dim
        # default settings
        # https://github.com/facebookresearch/dino/blob/main/main_dino.py#L64
        self.use_batchnorm = False
        # https://github.com/facebookresearch/dino/blob/main/main_dino.py#L57
        self.norm_last_layer = True
        # https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L257
        self.n_hidden_layers = 1

        super().__init__(**kwargs)
        self.register_buffer("center", torch.zeros(1, self.output_dim))
        # info for updating center (populated by DinoLoss)
        self.center_update_info = None
        self._reset_center_update_info()

    def _reset_center_update_info(self):
        self.center_update_info = dict(momentum=None, batch_centers=[])

    def after_update_step(self):
        super().after_update_step()
        with torch.no_grad():
            # update center for teacher output
            # update only after update_step (allows gradient accumulation)
            center_momentum = self.center_update_info["momentum"]
            batch_centers = self.center_update_info["batch_centers"]
            batch_center = torch.concat(batch_centers).mean(dim=0, keepdim=True)
            # noinspection PyAttributeOutsideInit
            self.center = self.center * center_momentum + batch_center * (1 - center_momentum)
        self._reset_center_update_info()

    @property
    def _requires_initializer(self):
        return False

    def _model_specific_initialization_before_ema(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # TODO batchnorm is not initialized (default setting doesn't use it)
        self.apply(_init_weights)
        self.last_layer.weight_g.data.fill_(1)

    def create_projector(self):
        linear_bias = not self.use_batchnorm
        norm = nn.BatchNorm1d if self.use_batchnorm else lambda _: nn.Identity()
        act = nn.GELU
        # initial layer
        layers = [
            nn.Linear(self.input_dim, self.proj_hidden_dim, bias=linear_bias),
            norm(self.proj_hidden_dim),
            act(),
        ]
        # hidden layers (1 by default)
        for _ in range(self.n_hidden_layers):
            layers += [
                nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim, bias=linear_bias),
                norm(self.proj_hidden_dim),
                act(),
            ]
        # bottleneck + norm + expansion
        self.last_layer = nn.utils.weight_norm(nn.Linear(self.bottleneck_dim, self.output_dim, bias=False))
        if self.norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        layers += [
            nn.Linear(self.proj_hidden_dim, self.bottleneck_dim),
            Normalize(dim=1, p=2),
            self.last_layer,
        ]
        return nn.Sequential(*layers)
