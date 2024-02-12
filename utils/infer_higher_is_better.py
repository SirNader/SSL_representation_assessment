LOWER_IS_BETTER_KEYS = [
    "loss",
    "eigenvalue_entropy",
    "uniformity",
]
HIGHER_IS_BETTER_KEYS = [
    "accuracy",
    "ap",
    "auroc",
    "auprc",
    "alignment",
    "eigenvalue_cdf",
    "bestf1",
]
NEUTRAL_KEYS = ["optim", "profiling", "mask_ratio", "freezers"]


def _get_first_key(key):
    assert "/" in key
    first_key = key.split("/")[0]
    return first_key


def is_neutral_key(metric_key):
    first_key = _get_first_key(metric_key)
    return first_key in NEUTRAL_KEYS


def higher_is_better_from_metric_key(metric_key):
    first_key = _get_first_key(metric_key)
    for key in HIGHER_IS_BETTER_KEYS:
        if key in first_key:
            return True
    for key in LOWER_IS_BETTER_KEYS:
        if key in first_key:
            return False
    raise NotImplementedError(f"no mapping from {metric_key} to higher_is_better")
