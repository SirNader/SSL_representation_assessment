import torch

from schedules import schedule_from_kwargs
from utils.model_utils import update_ema, copy_params
from .projector_head import ProjectorHead


class MomentumProjectorHead(ProjectorHead):
    def __init__(self, target_factor=None, target_factor_schedule=None, **kwargs):
        super().__init__(**kwargs)
        self.momentum_projector = self.create_projector()
        for param in self.momentum_projector.parameters():
            param.requires_grad = False

        # EMA schedule
        assert (target_factor is not None) ^ (target_factor_schedule is not None)
        self.target_factor = target_factor
        self.target_factor_schedule = schedule_from_kwargs(target_factor_schedule, update_counter=self.update_counter)

        # make sure to not overwrite copying ema parameters
        assert type(self)._model_specific_initialization == MomentumProjectorHead._model_specific_initialization

    def _model_specific_initialization(self):
        self.logger.info(f"initializing {type(self).__name__}.target_projector with parameters from projector")
        copy_params(self.projector, self.momentum_projector)
        self._model_specific_initialization_before_ema()

    def _model_specific_initialization_before_ema(self):
        pass

    def after_update_step(self):
        if self.target_factor_schedule is not None:
            target_factor = self.target_factor_schedule.get_value(self.update_counter.cur_checkpoint)
        else:
            target_factor = self.target_factor
        update_ema(self.projector, self.momentum_projector, target_factor)

    def forward(self, x):
        result = super().forward(x)
        if "momentum" in x:
            # local views don't need momentum forward
            with torch.no_grad():
                momentum_projector_input = self._prepare_forward(x["momentum"])
                result["momentum_projected"] = self.momentum_projector(momentum_projector_input)
        return result
