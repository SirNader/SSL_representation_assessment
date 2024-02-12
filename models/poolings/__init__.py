import torch.nn as nn

from models.poolings.single_pooling import SinglePooling
from utils.factory import instantiate


def pooling_from_kwargs(kind, model=None, **kwargs):
    try:
        return SinglePooling(kind, model=model, **kwargs)
    except NotImplementedError:
        assert model is not None
        return instantiate(module_names=[f"models.poolings.{kind}"], type_names=[kind], model=model, **kwargs)


def pooling2d_from_kwargs(kind, factor):
    if kind is None:
        return nn.Identity()
    if kind == "max":
        return nn.MaxPool2d(kernel_size=factor, stride=factor)
    if kind in ["mean", "avg", "average"]:
        return nn.AvgPool2d(kernel_size=factor, stride=factor)
    raise NotImplementedError
