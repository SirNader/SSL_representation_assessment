from models.patchcore.aggregators import aggregator_from_kwargs
from models.patchcore.finalizers import finalizer_from_kwargs
from models.patchcore.selectors import selector_from_kwargs
from utils.factory import create
from .base.pooling_base import PoolingBase
from .single_pooling import SinglePooling


class MultiPooling(PoolingBase):
    def __init__(self, aggregator, selector, pooling, finalizer, model):
        self.selector = create(selector, selector_from_kwargs)
        self.aggregator = create(aggregator, aggregator_from_kwargs, selectors=[self.selector])
        self.pooling = create(pooling, SinglePooling, model=model)
        self.finalizer = create(finalizer, finalizer_from_kwargs)

        self.aggregator.register_hooks(model)
        self.aggregator.disable_raise_exception()

    def __call__(self, x, *_, **__):
        features = {k: v for k, v in self.aggregator.outputs.items()}
        self.aggregator.outputs.clear()
        features = self.selector(features)
        features = [self.pooling(f) for f in features]
        features = self.finalizer(features)
        return features
