import pickle

import networkx as nx
import numpy as np
import tensorflow as tf

from chia import instrumentation, knowledge
from chia.components.classifiers import keras_hierarchicalclassification


class LabelSharingEmbeddingBasedKerasHC(
    keras_hierarchicalclassification.EmbeddingBasedKerasHC, instrumentation.Observable
):
    """Implementation of Fergus, R., Bernal, H., Weiss, Y., & Torralba, A. (2010, September).
    Semantic label sharing for learning with many categories. In European Conference on Computer Vision (pp. 762-775).
    Springer, Berlin, Heidelberg.
    """

    def __init__(self, kb, l2=5e-5, kappa=10.0):
        instrumentation.Observable.__init__(self)
        keras_hierarchicalclassification.EmbeddingBasedKerasHC.__init__(self, kb)

        # Configuration
        self._l2_regularization_coefficient = l2
        self._kappa = 10

        self.last_observed_concept_count = len(
            self.kb.concepts(flags={knowledge.ConceptFlag.PREDICTION_TARGET})
        )

        self.fc_layer = None
        self.uid_to_dimension = {}
        self.dimension_to_uid = []
        self.affinity_matrix = None

        self.update_embedding()

    def update_embedding(self):
        current_observed_concepts = self.kb.concepts(
            flags={knowledge.ConceptFlag.PREDICTION_TARGET}
        )
        current_observed_concept_count = len(current_observed_concepts)

        try:
            old_weights = self.fc_layer.get_weights()
        except Exception:
            old_weights = []

        self.fc_layer = tf.keras.layers.Dense(
            current_observed_concept_count,
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(
                self._l2_regularization_coefficient
            )
            if self._l2_regularization_coefficient > 0.0
            else None,
            kernel_initializer="zero",
            bias_initializer="zero",
        )

        update_uids = [
            concept.uid
            for concept in current_observed_concepts
            if concept.uid not in self.dimension_to_uid
        ]

        self.dimension_to_uid += sorted(update_uids)

        self.uid_to_dimension = {
            uid: dimension for dimension, uid in enumerate(self.dimension_to_uid)
        }

        if len(old_weights) == 2:
            # Layer can be updated
            new_weights = np.concatenate(
                [
                    old_weights[0],
                    np.zeros(
                        [
                            old_weights[0].shape[0],
                            current_observed_concept_count
                            - self.last_observed_concept_count,
                        ]
                    ),
                ],
                axis=1,
            )
            new_biases = np.concatenate(
                [
                    old_weights[1],
                    np.zeros(
                        current_observed_concept_count
                        - self.last_observed_concept_count
                    ),
                ],
                axis=0,
            )

            self.fc_layer.build([None, old_weights[0].shape[0]])

            self.fc_layer.set_weights([new_weights, new_biases])

        self._update_affinity_matrix()

        self.report_metric("observed_concepts", current_observed_concept_count)
        self.last_observed_concept_count = current_observed_concept_count

    def predict_embedded(self, feature_batch):
        return self.fc_layer(feature_batch)

    def embed(self, labels):
        embeddings = []
        for label in labels:
            if label == "chia::UNCERTAIN":
                embeddings += [
                    np.full(
                        self.last_observed_concept_count,
                        fill_value=1.0 / self.last_observed_concept_count,
                    )
                ]
            else:
                embeddings += [
                    tf.one_hot(
                        self.uid_to_dimension[label],
                        depth=self.last_observed_concept_count,
                    )
                ]

        embedded_labels = tf.stack(embeddings)

        # Apply affinity matrix to labels
        embedded_labels = embedded_labels @ self.affinity_matrix

        return embedded_labels

    def deembed_dist(self, embedded_labels):
        return [
            [
                (uid, embedded_label[dim] / embedded_label_sum)
                for uid, dim in self.uid_to_dimension.items()
            ]
            for (embedded_label, embedded_label_sum) in zip(
                embedded_labels, np.sum(embedded_labels, axis=1)
            )
        ]

    def loss(self, feature_batch, ground_truth, weight_batch, global_step):
        embedding = self.embed(ground_truth)
        prediction = self.predict_embedded(feature_batch)

        # Binary cross entropy loss function from keras_idk
        clipped_probs = tf.clip_by_value(prediction, 1e-7, (1.0 - 1e-7))
        the_loss = -(
            embedding * tf.math.log(clipped_probs)
            + (1.0 - embedding) * tf.math.log(1.0 - clipped_probs)
        )

        # We can't use tf's binary_crossentropy because it always takes a mean around axis -1,
        # but we need the sum
        sum_per_batch_element = tf.reduce_sum(the_loss, axis=1)
        return tf.reduce_mean(sum_per_batch_element * weight_batch)

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
            pickle.dump((self.uid_to_dimension, self.dimension_to_uid), target)

    def restore(self, path):
        self.maybe_update_embedding()
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
            (self.uid_to_dimension, self.dimension_to_uid) = pickle.load(target)

        self.update_embedding()

    def _update_affinity_matrix(self):
        if len(self.uid_to_dimension) == 0:
            self.affinity_matrix = None
            return

        # Number of dimensions
        dimensions = len(self.uid_to_dimension)
        self.affinity_matrix = np.zeros((dimensions, dimensions))

        graph: nx.DiGraph = self.kb.get_hyponymy_relation_rgraph()

        for i in range(dimensions):
            node_i = self.dimension_to_uid[i]
            parents_i = set(nx.ancestors(graph, node_i))

            for j in range(i, dimensions):
                node_j = self.dimension_to_uid[j]
                parents_j = set(nx.ancestors(graph, node_j))

                # Calculate S_ij
                numerator = len(parents_i.intersection(parents_j))
                denominator = max(len(parents_i), len(parents_j))

                if denominator > 0:
                    S_ij = float(numerator) / denominator
                else:
                    S_ij = 1.0

                if i == j:
                    assert S_ij == 1.0

                # Build matrix
                affinity_ij = np.exp(-self._kappa * (1.0 - S_ij))
                self.affinity_matrix[i, j] = affinity_ij
                self.affinity_matrix[j, i] = affinity_ij

        if not np.all(self.affinity_matrix > 0.0):
            raise ValueError("Affinity matrix contains zero entry!")

        if not np.all(self.affinity_matrix <= 1.0):
            raise ValueError("Affinity matrix entry greater than one!")
