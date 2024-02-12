import torch
import torch.nn as nn

from models import model_from_kwargs, remove_all_optims_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from models.ssl_heads.base.momentum_projector_head import MomentumProjectorHead
from models.ssl_heads.base.momentum_projector_predictor_head import MomentumProjectorPredictorHead
from schedules import schedule_from_kwargs
from utils.factory import create, create_collection
from utils.model_utils import update_ema, copy_params
from utils.multi_crop_utils import multi_crop_forward


class SslModel(CompositeModelBase):
    def __init__(
            self,
            backbone,
            heads,
            momentum_backbone=None,
            target_factor=None,
            target_factor_schedule=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = create(
            backbone,
            model_from_kwargs,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            stage_path_provider=self.stage_path_provider,
        )
        self.latent_shape = self.backbone.output_shape
        self.heads = create_collection(
            heads,
            model_from_kwargs,
            backbone=self.backbone,
            input_shape=self.backbone.output_shape,
            update_counter=self.update_counter,
            stage_path_provider=self.stage_path_provider,
        )
        self.heads = nn.ModuleDict(self.heads)

        # initialize backbone EMA if necessary
        self.target_factor = target_factor
        if (
                any(
                    isinstance(head, (MomentumProjectorPredictorHead, MomentumProjectorHead))
                    for head in self.heads.values()
                )
        ):
            assert (target_factor is not None) ^ (target_factor_schedule is not None)
            self.target_factor_schedule = schedule_from_kwargs(
                target_factor_schedule,
                update_counter=self.update_counter,
            )
            if isinstance(backbone, dict):
                momentum_backbone = remove_all_optims_from_kwargs(backbone)
                self.momentum_backbone = create(
                    momentum_backbone,
                    model_from_kwargs,
                    input_shape=self.input_shape,
                    update_counter=self.update_counter,
                    stage_path_provider=self.stage_path_provider,
                    is_frozen=True,
                )
            else:
                assert momentum_backbone is not None
                self.momentum_backbone = momentum_backbone
        else:
            assert momentum_backbone is None and target_factor is None and target_factor_schedule is None
            self.momentum_backbone = None
            self.target_factor_schedule = None

    @property
    def submodels(self):
        submodels = dict(backbone=self.backbone, **{f"heads.{key}": value for key, value in self.heads.items()})
        if self.momentum_backbone is not None:
            submodels["momentum_backbone"] = self.momentum_backbone
        return submodels

    # noinspection PyMethodOverriding
    def forward(self, x, mask_generator=None):
        forward_kwargs = dict(mask_generator=mask_generator) if mask_generator is not None else {}
        backbone_output = multi_crop_forward(model=self.backbone, x=x, **forward_kwargs)
        if isinstance(backbone_output, list) and not isinstance(backbone_output[0], dict):
            backbone_output = [dict(x=bbo) for bbo in backbone_output]

        if self.momentum_backbone is not None:
            # only propagate global views (first 2 views) through momentum backbone
            with torch.no_grad():
                momentum_bb_output = multi_crop_forward(model=self.backbone, x=x[:2], **forward_kwargs)
                if not isinstance(momentum_bb_output[0], dict):
                    momentum_bb_output = [dict(x=mbbo) for mbbo in momentum_bb_output]
                assert len(momentum_bb_output) <= len(backbone_output)
                for i in range(len(momentum_bb_output)):
                    if isinstance(backbone_output[i], dict):
                        backbone_output[i]["momentum"] = momentum_bb_output[i]
                    else:
                        backbone_output[i] = dict(x=backbone_output[i], momentum=momentum_bb_output[i])

        heads_output = {name: multi_crop_forward(model=head, x=backbone_output) for name, head in self.heads.items()}
        return dict(backbone_output=backbone_output, heads_output=heads_output)

    # region EMA initialization/update
    def _model_specific_initialization(self):
        if self.momentum_backbone is not None and self.momentum_backbone.should_apply_model_specific_initialization:
            self.logger.info(f"initializing momentum_backbone with parameters from backbone")
            copy_params(self.backbone, self.momentum_backbone)
        super()._model_specific_initialization()

    def after_update_step(self):
        if self.momentum_backbone is not None:
            if self.target_factor_schedule is not None:
                target_factor = self.target_factor_schedule.get_value(self.update_counter.cur_checkpoint)
            else:
                target_factor = self.target_factor
            update_ema(self.backbone, self.momentum_backbone, target_factor)
    # endregion
