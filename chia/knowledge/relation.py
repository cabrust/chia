import abc
import enum
import typing

import networkx as nx

from chia import instrumentation
from chia.knowledge.messages import RelationChangeMessage


class RelationSource(abc.ABC):
    @abc.abstractmethod
    def get_left_for(self, uid_right):
        pass

    @abc.abstractmethod
    def get_right_for(self, uid_left):
        pass


class StaticRelationSource:
    def __init__(self, data):
        self.right_for_left = {}
        self.left_for_right = {}
        for (left, right) in data:
            if left in self.right_for_left.keys():
                self.right_for_left[left] += [right]
            else:
                self.right_for_left[left] = [right]

            if right in self.left_for_right.keys():
                self.left_for_right[right] += [left]
            else:
                self.left_for_right[right] = [left]

    def get_left_for(self, uid_right):
        if uid_right in self.left_for_right.keys():
            return self.left_for_right[uid_right]
        else:
            return set()

    def get_right_for(self, uid_left):
        if uid_left in self.right_for_left.keys():
            return self.right_for_left[uid_left]
        else:
            return set()


class RelationFlag(enum.Enum):
    HYPONYMY = enum.auto()
    SYMMETRIC = enum.auto()
    REFLEXIVE = enum.auto()
    TRANSITIVE = enum.auto()
    EXPLORE_LEFT = enum.auto()
    EXPLORE_RIGHT = enum.auto()


class Relation(instrumentation.Observable):
    def __init__(
        self,
        uid: str,
        sources: typing.List[RelationSource],
        flags: typing.Optional[typing.Set[RelationFlag]] = None,
    ):
        instrumentation.Observable.__init__(self)
        self.uid: str = uid
        self.sources: typing.List[RelationSource] = sources

        if flags is not None:
            self.flags: typing.Set[RelationFlag] = flags
        else:
            self.flags: typing.Set[RelationFlag] = set()

        self._rgraph: nx.DiGraph = nx.DiGraph()
        self._ugraph: nx.Graph = nx.Graph()
        self._pairs: typing.Set[typing.Tuple[str, str]] = set()

    def pairs(self) -> typing.Set[typing.Tuple[str, str]]:
        return self._pairs

    def add_pairs(self, pairs: typing.Set[typing.Tuple[str, str]]):
        changed = False
        for pair in pairs:
            changed |= self._add_pair(pair)

        if changed:
            message = RelationChangeMessage(self._sender_name())
            self.notify(message)

    def _add_pair(self, pair: typing.Tuple[str, str]):
        if pair not in self._pairs:
            self._pairs |= {pair}

            # This is reversed on purpose
            self._rgraph.add_edge(pair[1], pair[0])
            self._ugraph.add_edge(pair[1], pair[0])
            return True
        else:
            return False

    def rgraph(self) -> nx.DiGraph:
        return self._rgraph

    def ugraph(self) -> nx.Graph:
        return self._ugraph
