import yaml

from utils.infer_higher_is_better import higher_is_better_from_metric_key, is_neutral_key
from .base.summary_provider_base import SummaryProviderBase
from ..stage_path_provider import StagePathProvider


class PrimitiveSummaryProvider(SummaryProviderBase):
    def __init__(self, stage_path_provider: StagePathProvider):
        super().__init__()
        self.stage_path_provider = stage_path_provider
        self.summary = {}

    def update(self, *args, **kwargs):
        self.summary.update(*args, **kwargs)

    def __setitem__(self, key, value):
        self.summary[key] = value

    def __getitem__(self, key):
        return self.summary[key]

    def __contains__(self, key):
        return key in self.summary

    def keys(self):
        return self.summary.keys()

    def get_summary_of_previous_stage(self, stage_name, stage_id):
        summary_uri = self.stage_path_provider.get_primitive_summary_uri(stage_name=stage_name, stage_id=stage_id)
        if not summary_uri.exists():
            return None

        with open(summary_uri) as f:
            return yaml.safe_load(f)

    def flush(self):
        """ summary is potentially often updated -> flush in bulks """
        with open(self.stage_path_provider.primitive_summary_uri, "w") as f:
            yaml.safe_dump(self.summary, f)

    def summarize_logvalues(self):
        entries_uri = self.stage_path_provider.primitive_entries_uri
        if not entries_uri.exists():
            return None
        with open(entries_uri) as f:
            entries = yaml.safe_load(f)
        if entries is None:
            return None
        minmax_dict = {}
        for key, update_to_value in entries.items():
            # logvalues are stored as {"key": {<update0>: <value0>, <update1>: <value1>}}
            # NOTE: python min/max is faster on dicts than numpy
            last_update = max(update_to_value.keys())
            last_value = update_to_value[last_update]
            self[key] = last_value

            if key in ["epoch", "update", "sample"]:
                continue
            # exclude neutral keys (e.g. lr, profiling, ...) for min/max summarizing
            if is_neutral_key(key):
                continue
            higher_is_better = higher_is_better_from_metric_key(key)
            if higher_is_better:
                minmax_key = f"{key}/max"
                minmax_value = max(update_to_value.values())
            else:
                minmax_key = f"{key}/min"
                minmax_value = min(update_to_value.values())
            self[minmax_key] = minmax_value
            minmax_dict[minmax_key] = minmax_value
            self.logger.info(f"{minmax_key}: {minmax_value}")
        return minmax_dict
