import pickle

import networkx as nx
import numpy as np
import tensorflow as tf

from chia import instrumentation, knowledge
from chia.components.classifiers import keras_hierarchicalclassification


class IDKEmbeddingBasedKerasHC(
    keras_hierarchicalclassification.EmbeddingBasedKerasHC, instrumentation.Observable
):
    def __init__(
        self,
        kb,
        l2=5e-5,
        mlnp=True,
        normalize_scores=True,
        raw_output=False,
        weighting="default",
        gain_compensation="simple",
    ):
        keras_hierarchicalclassification.EmbeddingBasedKerasHC.__init__(self, kb)
        instrumentation.Observable.__init__(self)

        # Configuration
        self._l2_regularization_coefficient = l2

        self._mlnp = mlnp
        self._normalize_scores = normalize_scores

        self._raw_output = raw_output
        if self._raw_output and (self._mlnp or self._normalize_scores):
            raise ValueError(
                "Cannot use raw output and MLNP or normalization at the same time!"
            )

        self._weighting = weighting
        self._gain_compensation = gain_compensation

        self.fc_layer = None
        self.uid_to_dimension = {}
        self.graph = None
        self.prediction_target_uids = None
        self.topo_sorted_uids = None
        self.loss_weights = None
        self.update_embedding()

    def predict_embedded(self, feature_batch):
        return self.fc_layer(feature_batch)

    def embed(self, labels):
        embedding = np.zeros((len(labels), len(self.uid_to_dimension)))
        for i, label in enumerate(labels):
            if label == "chia::UNCERTAIN":
                embedding[i] = 1.0
            else:
                embedding[i, self.uid_to_dimension[label]] = 1.0
                for ancestor in nx.ancestors(self.graph, label):
                    embedding[i, self.uid_to_dimension[ancestor]] = 1.0

        return embedding

    def deembed_dist(self, embedded_labels):
        return [
            self._deembed_single(embedded_label) for embedded_label in embedded_labels
        ]

    def _deembed_single(self, embedded_label):
        conditional_probabilities = {
            uid: embedded_label[i] for uid, i in self.uid_to_dimension.items()
        }

        if self._raw_output:
            # Directly output conditional probabilities
            return list(conditional_probabilities.items())
        else:
            # Stage 1 calculates the unconditional probabilities
            unconditional_probabilities = {}

            for uid in self.topo_sorted_uids:
                unconditional_probability = conditional_probabilities[uid]

                no_parent_probability = 1.0
                has_parents = False
                for parent in self.graph.predecessors(uid):
                    has_parents = True
                    no_parent_probability *= 1.0 - unconditional_probabilities[parent]

                if has_parents:
                    unconditional_probability *= 1.0 - no_parent_probability

                unconditional_probabilities[uid] = unconditional_probability

            # Stage 2 calculates the joint probability of the synset and "no children"
            joint_probabilities = {}
            for uid in reversed(self.topo_sorted_uids):
                joint_probability = unconditional_probabilities[uid]
                no_child_probability = 1.0
                for child in self.graph.successors(uid):
                    no_child_probability *= 1.0 - unconditional_probabilities[child]

                joint_probabilities[uid] = joint_probability * no_child_probability

            tuples = joint_probabilities.items()
            sorted_tuples = list(sorted(tuples, key=lambda tup: tup[1], reverse=True))

            if self._mlnp:
                for i, (uid, p) in enumerate(sorted_tuples):
                    if uid not in self.prediction_target_uids:
                        sorted_tuples[i] = (uid, 0.0)

            if self._normalize_scores:
                total_scores = sum([p for uid, p in sorted_tuples])
                sorted_tuples = [(uid, p / total_scores) for uid, p in sorted_tuples]

            return list(sorted_tuples)

    def update_embedding(self):
        current_concepts = self.kb.concepts()
        current_concept_count = len(current_concepts)
        self.report_metric("current_concepts", current_concept_count)

        if current_concept_count == 0:
            return True

        try:
            old_weights = self.fc_layer.get_weights()
            old_uidtodim = self.uid_to_dimension
            old_graph = self.graph

        except Exception:
            old_weights = []
            old_uidtodim = []
            old_graph = None

        self.fc_layer = tf.keras.layers.Dense(
            current_concept_count,
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(
                self._l2_regularization_coefficient
            )
            if self._l2_regularization_coefficient > 0.0
            else None,
            kernel_initializer="zero",
            bias_initializer="zero",
        )

        try:
            self.graph = self.kb.get_hyponymy_relation_rgraph()
        except ValueError:
            return False

        # Memorize topological sorting for later
        all_uids = nx.topological_sort(self.graph)
        self.topo_sorted_uids = list(all_uids)
        assert len(current_concepts) == len(self.topo_sorted_uids)

        self.uid_to_dimension = {
            uid: dimension for dimension, uid in enumerate(self.topo_sorted_uids)
        }

        self.prediction_target_uids = {
            concept.uid
            for concept in self.kb.concepts(
                flags={knowledge.ConceptFlag.PREDICTION_TARGET}
            )
        }

        if len(old_weights) == 2:
            # Layer can be updated
            new_weights = np.zeros([old_weights[0].shape[0], current_concept_count])
            new_biases = np.zeros([current_concept_count])

            reused_concepts = 0
            for new_uid, dim in self.uid_to_dimension.items():
                # Check if old weight is even available
                if new_uid not in old_uidtodim.keys():
                    continue

                # Check if parents have changed
                if set(self.graph.predecessors(new_uid)) != set(
                    old_graph.predecessors(new_uid)
                ):
                    continue

                new_weights[:, dim] = old_weights[0][:, old_uidtodim[new_uid]]
                new_biases[dim] = old_weights[1][old_uidtodim[new_uid]]
                reused_concepts += 1

            self.report_metric("reused_concepts", reused_concepts)

            self.fc_layer.build([None, old_weights[0].shape[0]])
            self.fc_layer.set_weights([new_weights, new_biases])

        self.update_loss_weights()
        return True

    def update_loss_weights(self):
        if len(self.prediction_target_uids) == 0:
            self.log_debug("Skipping loss weight update, no concepts found.")
            self.loss_weights = []
            return

        self.log_debug(
            f"Updating loss weights. Strategy: {self._weighting}, "
            f"gain compensation: {self._gain_compensation}"
        )

        # (1) Calculate "natural" weights by assuming uniform distribution
        # over observed concepts
        occurences = {uid: 0 for uid in self.topo_sorted_uids}
        for uid in self.prediction_target_uids:
            affected_uids = {uid}
            affected_uids |= nx.ancestors(self.graph, uid)
            for affected_uid in list(affected_uids):
                affected_uids |= set(self.graph.successors(affected_uid))

            for affected_uid in affected_uids:
                occurences[affected_uid] += 1

        occurrence_vector = np.array([occurences[uid] for uid in self.uid_to_dimension])

        # (2) Calculate weight vector
        if self._weighting == "default":
            self.loss_weights = np.ones(len(self.uid_to_dimension))

        elif self._weighting == "equalize":
            try:
                self.loss_weights = (
                    np.ones(len(self.uid_to_dimension)) / occurrence_vector
                )
            except ZeroDivisionError as err:
                self.log_fatal("Division by zero in equalize loss weighting strategy.")
                raise err

        elif self._weighting == "descendants":
            try:
                # Start with an equal weighting
                self.loss_weights = (
                    np.ones(len(self.uid_to_dimension)) / occurrence_vector
                )

                for i, uid in enumerate(self.uid_to_dimension):
                    self.loss_weights[i] *= (
                        len(nx.descendants(self.graph, uid)) + 1.0
                    )  # Add one for the node itself.
            except ZeroDivisionError as err:
                self.log_fatal(
                    "Division by zero in descendants loss weighting strategy."
                )
                raise err

        elif self._weighting == "reachable_leaf_nodes":
            try:
                # Start with an equal weighting
                self.loss_weights = (
                    np.ones(len(self.uid_to_dimension)) / occurrence_vector
                )

                for i, uid in enumerate(self.uid_to_dimension):
                    descendants = set(nx.descendants(self.graph, uid)) | {uid}
                    reachable_leaf_nodes = descendants.intersection(
                        self.prediction_target_uids
                    )
                    self.loss_weights[i] *= len(reachable_leaf_nodes)

                    # Test if any leaf nodes are reachable
                    if len(reachable_leaf_nodes) == 0:
                        raise ValueError(
                            f"In this hierarchy, the node {uid} cannot reach "
                            "any leaf nodes!"
                        )

            except ZeroDivisionError as err:
                self.log_fatal(
                    "Division by zero in reachable_leaf_nodes loss weighting strategy."
                )
                raise err

        else:
            raise ValueError(f'Unknown loss weighting strategy "{self._weighting}"')

        # Normalize so we don't have to adapt the learning rate a lot.
        if self._gain_compensation == "simple":
            gain = np.mean(self.loss_weights)
        elif self._gain_compensation == "per_element":
            gain = np.mean(self.loss_weights * occurrence_vector) / np.mean(
                occurrence_vector
            )
        else:
            raise ValueError(
                f'Unknown gain compensation setting "{self._gain_compensation}"'
            )

        self.report_metric("gain_from_weighting", gain)
        self.loss_weights /= gain

    def loss(self, feature_batch, ground_truth, global_step):
        if not self.is_updated:
            raise RuntimeError(
                "This classifier is not yet ready to compute a loss. "
                "Check if it has been notified of a hyponymy relation."
            )

        loss_mask = np.zeros((len(ground_truth), len(self.uid_to_dimension)))
        for i, label in enumerate(ground_truth):
            # Loss mask
            loss_mask[i, self.uid_to_dimension[label]] = 1.0

            for ancestor in nx.ancestors(self.graph, label):
                loss_mask[i, self.uid_to_dimension[ancestor]] = 1.0
                for successor in self.graph.successors(ancestor):
                    loss_mask[i, self.uid_to_dimension[successor]] = 1.0
                    # This should also cover the node itself, but we do it anyway

            if not self._mlnp:
                # Learn direct successors in order to "stop"
                # prediction at these nodes.
                # If MLNP is active, then this can be ignored.
                # Because we never want to predict
                # inner nodes, we interpret labels at
                # inner nodes as imprecise labels.
                for successor in self.graph.successors(label):
                    loss_mask[i, self.uid_to_dimension[successor]] = 1.0

        embedding = self.embed(ground_truth)
        prediction = self.predict_embedded(feature_batch)

        # Binary cross entropy loss function
        clipped_probs = tf.clip_by_value(prediction, 1e-7, (1.0 - 1e-7))
        the_loss = -(
            embedding * tf.math.log(clipped_probs)
            + (1.0 - embedding) * tf.math.log(1.0 - clipped_probs)
        )

        sum_per_batch_element = tf.reduce_sum(
            the_loss * loss_mask * self.loss_weights, axis=1
        )
        return tf.reduce_mean(sum_per_batch_element)

    def observe(self, samples, gt_resource_id):
        self.maybe_update_embedding()

    def regularization_losses(self):
        return self.fc_layer.losses

    def trainable_variables(self):
        return self.fc_layer.trainable_variables

    def save(self, path):
        with open(path + "_hc.pkl", "wb") as target:
            pickle.dump(self.fc_layer.get_weights(), target)

        with open(path + "_uidtodim.pkl", "wb") as target:
            pickle.dump((self.uid_to_dimension,), target)

    def restore(self, path):
        with open(path + "_hc.pkl", "rb") as target:
            new_weights = pickle.load(target)
            has_weights = False
            try:
                has_weights = len(self.fc_layer.get_weights()) == 2
            except Exception:
                pass

            if not has_weights:
                self.fc_layer.build([None, new_weights[0].shape[0]])

            self.fc_layer.set_weights(new_weights)

        with open(path + "_uidtodim.pkl", "rb") as target:
            (self.uid_to_dimension,) = pickle.load(target)

        self.update_embedding()
