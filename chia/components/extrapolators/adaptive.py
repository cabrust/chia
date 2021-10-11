import collections
import typing

import networkx as nx
import numpy as np

from chia.components.extrapolators.extrapolator import Extrapolator


class ICGainRangeExtrapolator(Extrapolator):
    def __init__(
        self,
        kb,
        apply_ground_truth,
        ic_method: typing.Optional[str] = None,
        ic_gain_target=0.1,
        ic_range=0.2,
        probability_threshold=0.55,
    ):
        super().__init__(
            kb=kb,
            apply_ground_truth=apply_ground_truth,
            ic_method=ic_method,
        )

        self._ic_gain_target = ic_gain_target
        self._ic_range = ic_range
        self._probability_threshold = probability_threshold

    def _extrapolate(self, ground_truth_uid, unconditional_probabilities):
        # We need this more often
        ground_truth_ic = self._ic_cache[ground_truth_uid]
        target_ic = ground_truth_ic + self._ic_gain_target
        half_range = self._ic_range / 2.0

        allowed_candidates = set(nx.descendants(self._rgraph, ground_truth_uid))

        candidates = [
            uid
            for (uid, probability) in unconditional_probabilities.items()
            if -half_range <= (self._ic_cache[uid] - target_ic) <= half_range
            and probability >= self._probability_threshold
            and uid in allowed_candidates
        ]

        if len(candidates) > 0:
            candidates_with_ic = [(uid, self._ic_cache[uid]) for uid in candidates]

            # Sort by probability first, see other methods for explanation of noise
            candidates_with_ic = list(
                sorted(
                    candidates_with_ic,
                    key=lambda x: unconditional_probabilities[x[0]]
                    + np.random.normal(0, 0.0001),
                    reverse=True,
                )
            )

            # Sort by IC. Stable sorting is guaranteed by python.
            candidates_with_ic = list(
                sorted(candidates_with_ic, key=lambda x: x[1], reverse=True)
            )
            return candidates_with_ic[0][0]
        else:
            return ground_truth_uid


class AdaptiveICGainExtrapolator(Extrapolator):
    def __init__(
        self,
        kb,
        apply_ground_truth,
        ic_method: typing.Optional[str] = None,
        ic_gain_target=0.1,
        min_threshold=0.55,
        max_threshold=1.0,
        learning_rate=1.0,
    ):
        super().__init__(
            kb=kb,
            apply_ground_truth=apply_ground_truth,
            ic_method=ic_method,
        )

        self._ic_gain_target = ic_gain_target
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._learning_rate = learning_rate

        # Initialize the threshold
        self._threshold = min_threshold

        self._last_ic_gains = collections.deque(maxlen=64)

    def _extrapolate(self, ground_truth_uid, unconditional_probabilities):
        """This is basically the same as SimpleThresholdCHILLAXExtrapolator, just with added reporting etc."""

        allowed_candidates = set(nx.descendants(self._rgraph, ground_truth_uid))

        candidates = [
            uid
            for (uid, probability) in unconditional_probabilities.items()
            if probability >= self._threshold and uid in allowed_candidates
        ]

        if len(candidates) > 0:
            candidates_with_ic = [(uid, self._ic_cache[uid]) for uid in candidates]

            # Sort by probability first, see other methods for explanation of noise
            candidates_with_ic = list(
                sorted(
                    candidates_with_ic,
                    key=lambda x: unconditional_probabilities[x[0]]
                    + np.random.normal(0, 0.0001),
                    reverse=True,
                )
            )

            # Sort by IC second. Stable sorting is guaranteed by python.
            candidates_with_ic = list(
                sorted(candidates_with_ic, key=lambda x: x[1], reverse=True)
            )
            return_value = candidates_with_ic[0][0]
        else:
            return_value = ground_truth_uid

        # Compute the actual IC gain of our actions
        realized_ic_gain = (
            self._ic_cache[return_value] - self._ic_cache[ground_truth_uid]
        )

        # Compute average IC gain, maxlen should do the rest :)
        self._last_ic_gains.append(realized_ic_gain)
        avg_ic_gain = sum(self._last_ic_gains) / float(len(self._last_ic_gains))

        # Assume that increasing the threshold decreases the possible IC gain
        # e.g. if average IC is 0.3 too much, increase the threshold by 0.3 (lr=1.0)
        step = self._learning_rate * (avg_ic_gain - self._ic_gain_target)
        self._threshold = max(
            self._min_threshold, min(self._max_threshold, self._threshold + step)
        )

        return return_value

    def reporting_report(self, current_step):
        """We want to have a look at the thresholds."""
        self.report_metric(
            "extrapolation_current_threshold", self._threshold, current_step
        )
        super().reporting_report(current_step)
