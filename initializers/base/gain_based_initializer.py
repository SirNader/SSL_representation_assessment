import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, BatchNorm2d, LayerNorm

from .initializer_base import InitializerBase


class GainBasedInitializer(InitializerBase):
    @property
    def should_apply_model_specific_initialization(self):
        return True

    @staticmethod
    def _get_gain(act_ctor):
        if act_ctor is None: return 1
        if act_ctor == nn.SELU: return 1

        # approximated
        if act_ctor == nn.ELU: return 1.245
        if act_ctor == nn.SiLU: return 1.676

        # from torch.nn.init.calculate_gain
        if act_ctor == nn.ReLU: return 2 ** 0.5
        if act_ctor == nn.Tanh: return 5 / 3

        # activations with constant parameter
        act = act_ctor()
        if isinstance(act, nn.LeakyReLU):
            # from torch.nn.init.calculate_gain
            return (2 / (1 + act.negative_slope ** 2)) ** 0.5

        approximation = GainBasedInitializer._approximate_gain(lambda x: act_ctor()(x))
        raise NotImplementedError(f"unknown gain factor for {act_ctor} (approximation is {approximation})")

    @staticmethod
    def _approximate_gain(fn):
        # https://github.com/pytorch/pytorch/issues/24991
        x = torch.randn(100000000)  # increase to get better approximation
        y = fn(x)
        return x.pow(2).mean().sqrt() / y.pow(2).mean().sqrt()

    def init_weights(self, model, **_):
        # TODO
        act_gain = 1  # GainBasedInitializer._get_gain(model.act_ctor)
        model.apply(lambda m: self._init(m, act_gain))
        self.logger.info(
            f"initialized {type(model).__name__} with "
            f"act_fn_gain={act_gain} and w_gain={type(self).__name__}"
        )

    @staticmethod
    def _get_weight_gain(weight):
        raise NotImplementedError

    @staticmethod
    def _get_fan_in_and_fan_out(weight):
        if len(weight.shape) == 1:
            return weight.size(0), weight.size(0)
        # noinspection PyProtectedMember
        return nn.init._calculate_fan_in_and_fan_out(weight)

    def _init(self, m, act_gain):
        # m.weight might is none e.g. for norm layers without affine
        if hasattr(m, "weight") and m.weight is not None:
            # explicitly add norms here to initialize with 1 (some norms might not want this)
            if type(m) in [BatchNorm1d, BatchNorm2d, LayerNorm]:
                nn.init.ones_(m.weight)
            else:
                cname = m.__class__.__name__
                if "norm" in cname.lower():
                    self.logger.warning(
                        f"{cname} looks like a norm layer but its GainBasedInitializer behavior is not defined"
                    )
                w_gain = self._get_weight_gain(m.weight)
                nn.init.normal_(m.weight, mean=0, std=act_gain * w_gain)

        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)
