from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    def update(self, samples, gt_resource_id, prediction_resource_id):
        return None

    @abstractmethod
    def result(self):
        return None

    @abstractmethod
    def reset(self):
        pass
