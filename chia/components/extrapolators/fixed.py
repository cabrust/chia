import typing

import networkx as nx
import numpy as np

from chia.components.extrapolators.extrapolator import Extrapolator


class SimpleThresholdExtrapolator(Extrapolator):
    def __init__(
        self,
        kb,
        apply_ground_truth,
        ic_method: typing.Optional[str] = None,
        threshold=0.55,
    ):
        super().__init__(
            kb=kb,
            apply_ground_truth=apply_ground_truth,
            ic_method=ic_method,
        )

        self._threshold = threshold

    def _extrapolate(self, ground_truth_uid, unconditional_probabilities):
        candidates = [
            uid
            for (uid, probability) in unconditional_probabilities.items()
            if probability >= self._threshold
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
            return candidates_with_ic[0][0]
        else:
            return ground_truth_uid


class DepthStepsCHILLAXExtrapolator(Extrapolator):
    def __init__(
        self,
        kb,
        apply_ground_truth,
        ic_method: typing.Optional[str] = None,
        steps=1,
        threshold=None,
    ):
        super().__init__(
            kb=kb,
            apply_ground_truth=apply_ground_truth,
            ic_method=ic_method,
        )

        self._steps = steps
        self._threshold = threshold

    def _extrapolate(self, ground_truth_uid, unconditional_probabilities):
        original_depth = self._uid_to_depth[ground_truth_uid]
        allowed_depth = original_depth + self._steps

        allowed_uids = set()
        for descendant in nx.descendants(self._rgraph, ground_truth_uid):
            if self._uid_to_depth[descendant] == allowed_depth:
                allowed_uids |= {descendant}
            elif (
                self._uid_to_depth[descendant] < allowed_depth
                and descendant in self._prediction_targets
            ):
                # We need to allow leaf nodes if they are shallower than the allowed depth.
                # Otherwise, we won't have any candidates sometimes.
                allowed_uids |= {descendant}

        candidates = [
            uid
            for (uid, probability) in unconditional_probabilities.items()
            if uid in allowed_uids
        ]

        if len(candidates) > 0:
            # When sorting by probability, add a very small amount of noise because of the nodes
            # that return exactly 0.5. Otherwise, the sorting is done alphabetically or topologically,
            # creating a bias.
            candidates = list(
                sorted(
                    candidates,
                    key=lambda x: unconditional_probabilities[x]
                    + np.random.normal(0, 0.0001),
                    reverse=True,
                )
            )
            if self._threshold is not None:
                candidates = [
                    candidate
                    for candidate in candidates
                    if unconditional_probabilities[candidate] > self._threshold
                ]
            if len(candidates) > 0:
                return candidates[0]
            else:
                return ground_truth_uid
        else:
            return ground_truth_uid
