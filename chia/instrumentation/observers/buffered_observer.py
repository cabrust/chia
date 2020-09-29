from chia.instrumentation.message import Message
from chia.instrumentation.observers.observer import Observer


class BufferedObserver(Observer):
    def __init__(self):
        self._messages = []

    def update(self, message: Message):
        self._messages += [message]

    def replay_messages(self, observable):
        for message in self._messages:
            observable.notify(message)
