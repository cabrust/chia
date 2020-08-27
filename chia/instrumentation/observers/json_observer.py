import gzip
import json
import logging
import pathlib

from chia.instrumentation.message import (
    ConfigMessage,
    LogMessage,
    Message,
    MetricMessage,
    ResultMessage,
    ShutdownMessage,
)
from chia.instrumentation.observers.observer import Observer


class JSONObserver(Observer):
    message_filter = (ConfigMessage, ResultMessage, MetricMessage, ShutdownMessage)
    log_message_filter = (LogMessage,)

    def __init__(
        self,
        experiment_metadata=None,
        environment_info=None,
        path_pattern="results/%s/%s.json.gz",
        log_level=logging.WARNING,
        compress=True,
    ):
        # Deal with potential nonexistant metadata
        if experiment_metadata is not None:
            experiment_name = experiment_metadata.name
            experiment_run_id = experiment_metadata.run_id
        else:
            experiment_name = "no_name"
            experiment_run_id = "no_run_id"

        # Try using the path pattern
        try:
            self.path = path_pattern % (
                experiment_name,
                experiment_run_id,
            )
        except TypeError:
            self.path = path_pattern

        self.log_level = log_level
        self.compress = compress

        # Make sure the directory exists
        parent_path = pathlib.Path(self.path).parent
        if not parent_path.exists():
            parent_path.mkdir(parents=True)

        # Try writing to path to fail early
        with open(self.path, "w") as f:
            f.write("{}")

        self.valid_messages = {
            filter_element.__name__: []
            for filter_element in list(self.message_filter)
            + list(self.log_message_filter)
        }

        self.valid_messages["metadata"] = self._object_to_dict(experiment_metadata)
        self.valid_messages["environment"] = self._object_to_dict(environment_info)

    def update(self, message: Message):
        if any(
            isinstance(message, filter_element)
            for filter_element in self.message_filter
        ):
            message_dict = self._object_to_dict(message)
            self.valid_messages[message.__class__.__name__] += [message_dict]
        elif any(
            isinstance(message, filter_element)
            for filter_element in self.log_message_filter
        ):
            message_log_level = message.level
            if message_log_level >= self.log_level:
                message_dict = self._object_to_dict(message)
                self.valid_messages[message.__class__.__name__] += [message_dict]

        if isinstance(message, ShutdownMessage):
            self.write_json()

    def write_json(self):
        if self.compress:
            with gzip.open(self.path, "wt", encoding="ascii") as f:
                json.dump(obj=self.valid_messages, fp=f)
        else:
            with open(self.path, "w") as f:
                json.dump(obj=self.valid_messages, fp=f, indent=2)

    @staticmethod
    def _object_to_dict(obj):
        if hasattr(obj, "__dict__"):
            ret_dict = dict()
            for key, value in obj.__dict__.items():
                try:
                    json.dumps({key: value})
                    ret_dict[key] = value
                except (ValueError, TypeError) as err:
                    ret_dict[key] = f"Could not serialize :( Error: {str(err)}"

            return ret_dict
        elif hasattr(obj, "_asdict"):
            return obj._asdict()
        else:
            return {"string_repr": str(obj)}
