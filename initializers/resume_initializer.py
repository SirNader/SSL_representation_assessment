import torch

from utils.checkpoint import Checkpoint
from .base.checkpoint_initializer import CheckpointInitializer


class ResumeInitializer(CheckpointInitializer):
    """
    initializes models/optims from a checkpoint ready for resuming training
    load_optim=True as this is usually used to resume a training run
    stage_name is provided by the trainer as it already knows the correct stage_name
    """

    def __init__(self, load_optim=True, **kwargs):
        super().__init__(load_optim=load_optim, **kwargs)
        assert isinstance(self.checkpoint, Checkpoint), "ResumeInitializer requires Epoch/Update/Sample checkpoint"

    def init_weights(self, model, **_):
        if model.is_frozen:
            self.logger.info(f"skip loading weights from checkpoint '{self.checkpoint}' for {model.name} (is_frozen)")
        else:
            self._init_weights(model)

    def _copy_config_and_summary(self, *_, **__):
        # resume initializer has the same config/summary as the new run
        pass

    def init_trainer(self, trainer):
        # LEGACY: checkpoints before 27.10.2022 don't have a trainer checkpoint
        try:
            ckpt_uri = self._get_ckpt_uri(prefix=f"trainer cp=", suffix=".th")
        except FileNotFoundError:
            self.logger.warning(f"no trainer checkpoint found for checkpoint {checkpoint}")
            return
        trainer.load_state_dict(torch.load(ckpt_uri))
