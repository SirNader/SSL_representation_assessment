from .base.gain_based_initializer import GainBasedInitializer


class XavierInitializer(GainBasedInitializer):
    @staticmethod
    def _get_weight_gain(weight):
        fan_in, fan_out = GainBasedInitializer._get_fan_in_and_fan_out(weight)
        mean_fan = (fan_in + fan_out) / 2
        return 1 / (mean_fan ** 0.5)
