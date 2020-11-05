import abc

from chia import instrumentation
from chia.knowledge import knowledge_base


class SampleTransformer(abc.ABC, instrumentation.Observable):
    def __init__(self, kb: knowledge_base.KnowledgeBase):
        instrumentation.Observable.__init__(self)
        self.kb = kb

    @abc.abstractmethod
    def transform(self, samples, is_training: bool, label_resource_id: str):
        pass
