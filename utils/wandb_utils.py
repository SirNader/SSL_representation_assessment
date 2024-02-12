import logging
from copy import deepcopy

import torch
import wandb

from configs.wandb_config import WandbConfig
from distributed.config import is_rank0, get_world_size, get_nodes
from providers.config_providers.noop_config_provider import NoopConfigProvider
from providers.config_providers.primitive_config_provider import PrimitiveConfigProvider
from providers.config_providers.wandb_config_provider import WandbConfigProvider
from providers.stage_path_provider import StagePathProvider
from providers.summary_providers.noop_summary_provider import NoopSummaryProvider
from providers.summary_providers.primitive_summary_provider import PrimitiveSummaryProvider
from providers.summary_providers.wandb_summary_provider import WandbSummaryProvider
from utils.kappaconfig.util import remove_large_collections


def init_wandb(
        device: str,
        run_name: str,
        stage_hp: dict,
        wandb_config: WandbConfig,
        stage_path_provider: StagePathProvider,
):
    logging.info("------------------")
    logging.info(f"initializing wandb (mode={wandb_config.mode})")
    # os.environ["WANDB_SILENT"] = "true"

    # create config_provider & summary_provider
    if not is_rank0():
        config_provider = NoopConfigProvider()
        summary_provider = NoopSummaryProvider()
        return config_provider, summary_provider
    elif wandb_config.is_disabled:
        config_provider = PrimitiveConfigProvider(stage_path_provider=stage_path_provider)
        summary_provider = PrimitiveSummaryProvider(stage_path_provider=stage_path_provider)
    else:
        config_provider = WandbConfigProvider(stage_path_provider=stage_path_provider)
        summary_provider = WandbSummaryProvider(stage_path_provider=stage_path_provider)

    config = {
        "run_name": run_name,
        "stage_name": stage_path_provider.stage_name,
        **_lists_to_dict(remove_large_collections(stage_hp)),
    }
    if not wandb_config.is_disabled:
        wandb.login(host=wandb_config.host)
        logging.info(f"logged into wandb (host={wandb_config.host})")
        name = run_name or "None"
        if stage_path_provider.stage_name != "default_stage":
            name += f"/{stage_path_provider.stage_name}"
        wandb.init(
            entity=wandb_config.entity,
            project=wandb_config.project,
            name=name,
            dir=str(stage_path_provider.stage_output_path),
            save_code=False,
            config=config,
            mode=wandb_config.mode,
            id=stage_path_provider.stage_id,
        )
    config_provider.update(config)

    # log additional environment properties
    additional_config = {}
    if str(device) == "cpu":
        additional_config["device"] = "cpu"
    else:
        additional_config["device"] = torch.cuda.get_device_name(0)
    additional_config["dist/world_size"] = get_world_size()
    additional_config["dist/nodes"] = get_nodes()
    config_provider.update(additional_config)

    return config_provider, summary_provider


def _lists_to_dict(root):
    """ wandb cant handle lists in configs -> transform lists into dicts with str(i) as key """
    #  (it will be displayed as [{"kind": "..."}, ...])
    root = deepcopy(root)
    return _lists_to_dicts_impl(dict(root=root))["root"]


def _lists_to_dicts_impl(root):
    if not isinstance(root, dict):
        return
    for k, v in root.items():
        if isinstance(v, list):
            root[k] = {str(i): vitem for i, vitem in enumerate(v)}
        elif isinstance(v, dict):
            root[k] = _lists_to_dicts_impl(root[k])
    return root


def finish_wandb(wandb_config: WandbConfig):
    if not is_rank0() or wandb_config.is_disabled:
        return
    wandb.finish()
