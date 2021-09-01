import abc
import typing

import networkx as nx
import numpy as np

from chia import instrumentation, knowledge


class Extrapolator(instrumentation.Observer, instrumentation.Observable, abc.ABC):
    def __init__(
        self,
        kb: knowledge.KnowledgeBase,
        apply_ground_truth: bool = False,
        ic_method: typing.Optional[str] = None,
    ):
        instrumentation.Observable.__init__(self)
        self.knowledge_base = kb
        self.knowledge_base.register(self)

        self.apply_ground_truth = apply_ground_truth

        self.is_updated = False

        # Graph Cache
        self._rgraph = nx.DiGraph()
        self._uid_to_depth = dict()
        self._prediction_targets = set()

        # Information Content Cache
        self._ic_calc: knowledge.InformationContentCalculator = (
            knowledge.InformationContentCalculatorFactory.create(
                {"name": ic_method if ic_method is not None else "zhou_2008_modified"}
            )
        )
        self._ic_cache = dict()

        self.update_relations_and_concepts()

        # Reporting
        self._reporting_samples_total = 0
        self._reporting_samples_changed = 0
        self._reporting_cum_ic_gain = 0

    def extrapolate(self, extrapolator_inputs):
        if not self.is_updated:
            raise RuntimeError(
                "This extrapolator is not updated. "
                "Please check if it is subscribed to "
                "RelationChange and ConceptChange messages."
            )

        outputs = []
        for ground_truth_uid, unconditional_probabilities in extrapolator_inputs:
            outputs += [
                self._extrapolate(ground_truth_uid, unconditional_probabilities)
            ]

        self._reporting_update(
            zip([gt_uid for gt_uid, _ in extrapolator_inputs], outputs)
        )
        return outputs

    def _reporting_update(self, label_pairs):
        for gt_uid, ext_uid in label_pairs:
            if gt_uid != ext_uid:
                self._reporting_samples_changed += 1
                self._reporting_cum_ic_gain += (
                    self._ic_cache[ext_uid] - self._ic_cache[gt_uid]
                )

            self._reporting_samples_total += 1

    def _reporting_reset(self):
        self._reporting_samples_total = 0
        self._reporting_samples_changed = 0
        self._reporting_cum_ic_gain = 0

    def reporting_report(self, current_step):
        if self._reporting_samples_total == 0:
            return

        self.report_metric(
            "extrapolation_changed_sample_fraction",
            self._reporting_samples_changed / float(self._reporting_samples_total),
            step=current_step,
        )
        self.report_metric(
            "extrapolation_avg_ic_gain",
            self._reporting_cum_ic_gain / float(self._reporting_samples_total),
            step=current_step,
        )

        self._reporting_reset()

    def update(self, message: instrumentation.Message):
        if isinstance(message, knowledge.RelationChangeMessage) or isinstance(
            message, knowledge.ConceptChangeMessage
        ):
            self.is_updated = False
            self.update_relations_and_concepts()

    def update_relations_and_concepts(self):
        try:
            # Update Information Content Cache
            self._ic_cache = dict()
            rgraph = self.knowledge_base.get_hyponymy_relation_rgraph()
            for concept in self.knowledge_base.concepts():
                self._ic_cache[
                    concept.uid
                ] = self._ic_calc.calculate_information_content(concept.uid, rgraph)

            # Graph Update
            self._rgraph = self.knowledge_base.get_hyponymy_relation_rgraph()
            self._prediction_targets = {
                concept.uid
                for concept in self.knowledge_base.concepts(
                    flags={knowledge.ConceptFlag.PREDICTION_TARGET}
                )
            }

            root = list(nx.topological_sort(self._rgraph))[0]
            self._uid_to_depth = {
                concept.uid: len(nx.shortest_path(self._rgraph, root, concept.uid))
                for concept in self.knowledge_base.concepts()
            }

        except ValueError as verr:
            self.log_warning(f"Could not update extrapolator. {verr.args}")

        self._update_relations_and_concepts()

    @abc.abstractmethod
    def _extrapolate(self, ground_truth_uid, unconditional_probabilities):
        pass

    def _update_relations_and_concepts(self):
        self.is_updated = True


class DoNothingExtrapolator(Extrapolator):
    def _extrapolate(self, ground_truth_uid, unconditional_probabilities):
        return ground_truth_uid


class ForcePredictionTargetExtrapolator(Extrapolator):
    def _extrapolate(self, ground_truth_uid, unconditional_probabilities):
        candidates = [
            uid
            for (uid, probability) in unconditional_probabilities.items()
            if uid in self._prediction_targets
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
            return candidates[0]
        else:
            return ground_truth_uid
