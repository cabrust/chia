import typing
from logging import DEBUG, ERROR, FATAL, INFO, WARNING

from chia.instrumentation.message import (
    LogMessage,
    Message,
    MetricMessage,
    ResultMessage,
    ShutdownMessage,
)
from chia.instrumentation.observers.observer import Observer


class Observable:
    def __init__(self):
        self._observers: typing.List[Observer] = list()

    def _sender_name(self):
        return self.__class__.__name__

    def register(self, observer: Observer):
        self._observers += [observer]

    def unregister(self, observer: Observer):
        self._observers.remove(observer)

    def notify(self, message: Message):
        for observer in self._observers:
            observer.update(message)

    # Helper functions for specific message types
    # Logging
    def log(self, level: int, message: str):
        self.notify(LogMessage(self._sender_name(), level, message))

    def log_debug(self, message: str):
        self.log(DEBUG, message)

    def log_info(self, message: str):
        self.log(INFO, message)

    def log_warning(self, message: str):
        self.log(WARNING, message)

    def log_error(self, message: str):
        self.log(ERROR, message)

    def log_fatal(self, message: str):
        self.log(FATAL, message)

    # Metrics
    def report_metric(self, metric: str, value: float, step: int = -1):
        self.notify(MetricMessage(self._sender_name(), metric, value, step))

    # Results
    def report_result(self, result_dict: dict, step: int = -1):
        self.notify(ResultMessage(self._sender_name(), result_dict, step))

    # Shutdown
    def send_shutdown(self, successful):
        self.notify(ShutdownMessage(self._sender_name(), successful=successful))


class NamedObservable(Observable):
    def __init__(self, custom_sender_name: str):
        super().__init__()
        self._custom_sender_name = custom_sender_name

    def _sender_name(self):
        return self._custom_sender_name
