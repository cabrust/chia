from abc import ABC, abstractmethod

from chia import knowledge


class Interactor(ABC):
    def __init__(self, kb: knowledge.KnowledgeBase):
        self._kb = kb

    @abstractmethod
    def query_annotations_for(self, samples, gt_resource_id, ann_resource_id):
        return None
