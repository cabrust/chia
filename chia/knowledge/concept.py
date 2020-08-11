import enum
import typing


class ConceptFlag(enum.Enum):
    PREDICTION_TARGET = enum.auto()
    AUTO_DISCOVERED = enum.auto()
    OBSERVED = enum.auto()


class Concept:
    def __init__(self, uid: str, flags: typing.Set[ConceptFlag] = None):
        self.uid = uid
        if flags is not None:
            self.flags = flags
        else:
            self.flags = set()
