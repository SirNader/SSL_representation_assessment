from providers.stage_path_provider import StagePathProvider
from .base.checkpoint_initializer import CheckpointInitializer


class PreviousStageInitializer(CheckpointInitializer):
    """
    initializes a model from a checkpoint of a previous run (specified by the stage_id)
    load_optim=False as this is usually used for frozen models
    """

    def __init__(self, stage_name, stage_path_provider: StagePathProvider, load_optim=False, **kwargs):
        # stage_id is from a previous stage
        assert stage_name in stage_path_provider.previous_stage_ids, f"invalid previous stage_name {stage_name}"
        stage_id = stage_path_provider.previous_stage_ids[stage_name]
        super().__init__(
            stage_name=stage_name,
            stage_id=stage_id,
            load_optim=load_optim,
            stage_path_provider=stage_path_provider,
            **kwargs,
        )
