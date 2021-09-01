from typing import Optional

import networkx as nx
import numpy as np

from chia import instrumentation, knowledge
from chia.components.interactors import interactor


class NoisyOracleInteractor(
    interactor.Interactor, instrumentation.Observable, instrumentation.Observer
):
    def __init__(
        self,
        kb,
        noise_model,
        inaccuracy=0.0,
        relabel_fraction=None,
        lambda_=None,
        q=None,
        filter_imprecise=False,
        project_to_random_leaf=False,
    ):
        interactor.Interactor.__init__(self, kb=kb)
        instrumentation.Observable.__init__(self)
        instrumentation.Observer.__init__(self)

        self.noise_model = noise_model
        self.inaccuracy = inaccuracy

        if self.noise_model == "Deng2014":
            assert relabel_fraction is not None
            self.relabel_fraction: float = relabel_fraction
        elif self.noise_model == "Poisson":
            assert lambda_ is not None
            self.lambda_: float = lambda_
        elif self.noise_model == "Geometric":
            assert q is not None
            self.q: float = q
        elif self.noise_model == "Inaccuracy":
            pass
        else:
            raise ValueError(f"Unknown noise model: {self.noise_model}")

        self.filter_imprecise = filter_imprecise
        self.project_to_random_leaf = project_to_random_leaf

        self.is_updated = False
        self.graph: Optional[nx.DiGraph] = None
        self.root = None
        self.leaf_nodes = None

        self._kb.register(self)

    def _apply_deng_noise(self, uid):
        if np.random.binomial(1, self.relabel_fraction):
            chosen_predecessor = next(
                self.graph.predecessors(uid)
            )  # TODO what to do if there is more than 1 parent?
            return chosen_predecessor
        else:
            return uid

    def _apply_geometric_noise(self, uid):
        target = np.random.geometric(1 - self.q) - 1
        return self._reduce_depth_to(uid, target)

    def _apply_poisson_noise(self, uid):
        target = np.random.poisson(self.lambda_)
        return self._reduce_depth_to(uid, target)

    def _reduce_depth_to(self, uid, depth_target):
        path_to_label = nx.shortest_path(self.graph, self.root, uid)
        final_depth = max(0, min(len(path_to_label) - 1, depth_target))
        return path_to_label[final_depth]

    def _project_to_random_leaf(self, uid):
        if self.graph.out_degree(uid) == 0:  # noqa
            return uid
        else:
            # List all descendants
            all_descendants = nx.descendants(self.graph, uid)

            # Use only leaves
            valid_descendants = list(
                filter(lambda n: self.graph.out_degree(n) == 0, all_descendants)  # noqa
            )

            return np.random.choice(valid_descendants)

    def _maybe_update_graphs(self):
        if not self.is_updated:
            try:
                self.graph = self._kb.get_hyponymy_relation_rgraph()
                self.root = next(nx.topological_sort(self.graph))
                self.leaf_nodes = list(
                    filter(
                        lambda n: self.graph.out_degree(n) == 0, self.graph.nodes
                    )  # noqa
                )
                self.is_updated = True
            except ValueError:
                # No graph available yet
                pass

    def query_annotations_for(self, samples, gt_resource_id, ann_resource_id):
        self._maybe_update_graphs()

        # Add noise
        noisy_samples = [
            sample.add_resource(
                self.__class__.__name__,
                ann_resource_id,
                self.apply_noise(sample.get_resource(gt_resource_id)),
            )
            for sample in samples
        ]

        # Count modified samples
        modified_samples = sum(
            [
                1
                if noisy_sample.get_resource(gt_resource_id)
                != noisy_sample.get_resource(ann_resource_id)
                else 0
                for noisy_sample in noisy_samples
            ]
        )
        self.log_debug(f"Modified {modified_samples} out of {len(samples)} samples.")

        # Filter imprecise samples
        precise_only_samples = [
            sample
            for sample in noisy_samples
            if self.apply_filter(sample.get_resource(ann_resource_id))
        ]
        self.log_debug(
            f"Filtered out {len(samples)-len(precise_only_samples)}"
            + f" out of {len(samples)} samples."
        )
        return precise_only_samples

    def apply_noise(self, uid):
        # Apply inaccuracy
        if np.random.uniform() <= self.inaccuracy:
            assert uid in self.leaf_nodes
            inaccurate_uid = np.random.choice(self.leaf_nodes)
        else:
            inaccurate_uid = uid

        # Select noise model
        if self.noise_model == "Deng2014":
            noisy_uid = self._apply_deng_noise(inaccurate_uid)
        elif self.noise_model == "Geometric":
            noisy_uid = self._apply_geometric_noise(inaccurate_uid)
        elif self.noise_model == "Poisson":
            noisy_uid = self._apply_poisson_noise(inaccurate_uid)
        elif self.noise_model == "Inaccuracy":
            noisy_uid = inaccurate_uid
        else:
            raise ValueError(f"Unknown noise model {self.noise_model}")

        # Project to random leaf
        if self.project_to_random_leaf:
            noisy_uid = self._project_to_random_leaf(noisy_uid)

        return noisy_uid

    def apply_filter(self, uid):
        if self.filter_imprecise:
            return self.graph.out_degree(uid) == 0  # noqa
        else:
            return True

    def update(self, message: instrumentation.Message):
        if isinstance(message, knowledge.RelationChangeMessage) or isinstance(
            message, knowledge.ConceptChangeMessage
        ):
            self.is_updated = False
