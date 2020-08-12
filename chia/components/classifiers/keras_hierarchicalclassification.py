from abc import ABC, abstractmethod

import chia.knowledge.messages
from chia import instrumentation, knowledge


class KerasHierarchicalClassifier(ABC):
    @abstractmethod
    def predict_dist(self, feature_batch):
        pass

    @abstractmethod
    def loss(self, feature_batch, ground_truth, global_step):
        pass

    @abstractmethod
    def observe(self, samples, gt_resource_id):
        pass

    @abstractmethod
    def regularization_losses(self):
        pass

    @abstractmethod
    def trainable_variables(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def restore(self, path):
        pass


class EmbeddingBasedKerasHC(KerasHierarchicalClassifier, instrumentation.Observer, ABC):
    def __init__(self, kb: knowledge.KnowledgeBase):
        self.kb = kb
        self.kb.register(self)

        self.is_updated = False

    @abstractmethod
    def predict_embedded(self, feature_batch):
        pass

    @abstractmethod
    def embed(self, labels):
        pass

    @abstractmethod
    def deembed_dist(self, embedded_labels):
        pass

    @abstractmethod
    def update_embedding(self):
        pass

    def predict_dist(self, feature_batch):
        embedded_predictions = self.predict_embedded(feature_batch).numpy()
        return self.deembed_dist(embedded_predictions)

    def maybe_update_embedding(self):
        if not self.is_updated:
            self.is_updated = self.update_embedding()

    def update(self, message: instrumentation.Message):
        if isinstance(
            message, chia.knowledge.messages.RelationChangeMessage
        ) or isinstance(message, chia.knowledge.messages.ConceptChangeMessage):
            self.is_updated = False
