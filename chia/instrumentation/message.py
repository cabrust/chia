import logging
import time
import typing


class Message:
    def __init__(self, sender: str):
        self.sender = sender
        self.timestamp: float = time.time()

    def _format_timestamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))

    def __str__(self):
        return f"[{self._format_timestamp()}] [MESSAGE] [{self.sender}]: {self.__class__.__name__}"


class LogMessage(Message):
    def __init__(self, sender: str, level: int, message: str):
        super().__init__(sender=sender)
        self.sender = sender
        self.level = level
        self.message = message

    def __str__(self):
        return (
            f"[{self._format_timestamp()}] [{logging.getLevelName(self.level)}] "
            f"[{self.sender}]: {self.message}"
        )


class ConfigMessage(Message):
    def __init__(self, sender: str, field: str, value: typing.Any, source: str):
        super().__init__(sender=sender)
        self.field = field
        self.value = value
        self.source = source

    def __str__(self):
        return (
            f"[{self._format_timestamp()}] [CONFIGURATION] [{self.sender}]: "
            f"config field {self.field} set to {self.value} from {self.source}"
        )


class MetricMessage(Message):
    def __init__(self, sender: str, metric: str, value: float, step: int):
        super().__init__(sender=sender)
        self.metric = metric
        self.value = value
        self.step = step

    def __str__(self):
        return (
            f"[{self._format_timestamp()}] [METRIC] [{self.sender}]: "
            f"metric {self.metric} @{self.step} = {self.value}"
        )


class ResultMessage(Message):
    def __init__(self, sender: str, result_dict: dict, step: int):
        super().__init__(sender=sender)
        self.result_dict = result_dict
        self.step = step

    def __str__(self):
        return (
            f"[{self._format_timestamp()}] [RESULT] [{self.sender}]: "
            f"keys are {','.join(self.result_dict.keys())} @{self.step}"
        )


class ShutdownMessage(Message):
    """This message tells the observers that they should save their data."""

    def __init__(self, sender: str, successful: bool):
        super().__init__(sender)
        self.successful = successful

    def __str__(self):
        return (
            f"[{self._format_timestamp()}] [SHUTDOWN] [{self.sender}] "
            f"Successful: {self.successful}"
        )
