import pickle

import numpy as np
import tensorflow as tf

from chia import instrumentation, knowledge
from chia.components.classifiers.keras_hierarchicalclassification import (
    EmbeddingBasedKerasHC,
)


class OneHotEmbeddingBasedKerasHC(EmbeddingBasedKerasHC, instrumentation.Observable):
    def __init__(self, kb, l2=5e-5):
        instrumentation.Observable.__init__(self)
        EmbeddingBasedKerasHC.__init__(self, kb)

        # Configuration
        self._l2_regularization_coefficient = l2

        self.last_observed_concept_count = len(
            self.kb.concepts(flags={knowledge.ConceptFlag.PREDICTION_TARGET})
        )

        self.fc_layer = None
        self.uid_to_dimension = {}
        self.dimension_to_uid = []

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
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(
                self._l2_regularization_coefficient
            )
            if self._l2_regularization_coefficient > 0.0
            else None,
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

        return tf.stack(embeddings)

    def deembed_dist(self, embedded_labels):
        return [
            [(uid, embedded_label[dim]) for uid, dim in self.uid_to_dimension.items()]
            for embedded_label in embedded_labels
        ]

    def loss(self, feature_batch, ground_truth, global_step):
        embedded_predictions = self.predict_embedded(feature_batch)
        embedded_ground_truth = self.embed(ground_truth)
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                embedded_ground_truth, embedded_predictions
            )
        )
        return loss

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
