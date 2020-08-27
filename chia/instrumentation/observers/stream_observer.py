import sys

from chia.instrumentation.message import Message
from chia.instrumentation.observers.observer import Observer


class StreamObserver(Observer):
    def __init__(self, stream=sys.stdout):
        self.stream = stream

    def update(self, message: Message):
        print(message, file=self.stream, flush=True)
