import abc

from chia.instrumentation.message import Message


class Observer(abc.ABC):
    @abc.abstractmethod
    def update(self, message: Message):
        pass
