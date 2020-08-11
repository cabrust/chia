import abc

from chia import knowledge


class Dataset(abc.ABC):
    @abc.abstractmethod
    def setup(self, **kwargs):
        pass

    def setups(self):
        return [dict()]

    @abc.abstractmethod
    def train_pool_count(self):
        pass

    @abc.abstractmethod
    def test_pool_count(self):
        pass

    @abc.abstractmethod
    def train_pool(self, index, label_resource_id):
        pass

    @abc.abstractmethod
    def test_pool(self, index, label_resource_id):
        pass

    @abc.abstractmethod
    def namespace(self):
        pass

    @abc.abstractmethod
    def get_hyponymy_relation_source(self) -> knowledge.RelationSource:
        pass

    @abc.abstractmethod
    def prediction_targets(self):
        pass
