from .base.gain_based_initializer import GainBasedInitializer


class HeInitializer(GainBasedInitializer):
    @staticmethod
    def _get_weight_gain(weight):
        fan_in, _ = GainBasedInitializer._get_fan_in_and_fan_out(weight)
        return 1 / (fan_in ** 0.5)
