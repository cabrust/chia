from chia import components
from chia.components.extrapolators import adaptive, fixed
from chia.components.extrapolators.extrapolator import (
    DoNothingExtrapolator,
    Extrapolator,
    ForcePredictionTargetExtrapolator,
)


class ExtrapolatorFactory(components.Factory):
    name_to_class_mapping = {
        "do_nothing": DoNothingExtrapolator,
        "simple_threshold": fixed.SimpleThresholdExtrapolator,
        "depth_steps": fixed.DepthStepsCHILLAXExtrapolator,
        "force_prediction_target": ForcePredictionTargetExtrapolator,
        "adaptive_ic_gain": adaptive.AdaptiveICGainExtrapolator,
        "ic_gain_range": adaptive.ICGainRangeExtrapolator,
    }


__all__ = ["ExtrapolatorFactory", "Extrapolator"]
