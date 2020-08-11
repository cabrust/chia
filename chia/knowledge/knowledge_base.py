import typing

from chia import components, instrumentation
from chia.knowledge.concept import Concept, ConceptFlag
from chia.knowledge.messages import ConceptChangeMessage, RelationChangeMessage
from chia.knowledge.relation import Relation, RelationFlag, RelationSource


class KnowledgeBase(instrumentation.Observable, instrumentation.Observer):
    def __init__(self):
        instrumentation.Observable.__init__(self)
        self._concepts: typing.Dict[str, Concept] = dict()
        self._relations: typing.Dict[str, Relation] = dict()

        # Build discovery object
        from chia.knowledge.discovery import Discovery

        self._discovery = Discovery(self)
        self.register(self._discovery)

    def add_relation(self, relation: Relation):
        if relation.uid in self._relations.keys():
            raise ValueError(f"Attempted to add relation {relation.uid} twice!")
        else:
            self._relations[relation.uid] = relation

            relation.register(self)

            message = RelationChangeMessage(self._sender_name())
            self.notify(message)

    def add_concepts(self, concepts: typing.List[Concept]):
        changed = False
        for concept in concepts:
            changed |= self._add_concept(concept)

        if changed:
            message = ConceptChangeMessage(self._sender_name())
            self.notify(message)

    def _add_concept(self, concept: Concept) -> bool:
        """Adds a concept to the knowledge base or merges the flags.

        Returns: True if the knowledge base has been changed in any way."""
        if concept.uid in self._concepts.keys():
            if concept.flags != self._concepts[concept.uid].flags:
                # Have to update flags
                self._concepts[concept.uid].flags |= concept.flags
                return True
            else:
                # Nothing to do
                return False
        else:
            self._concepts[concept.uid] = concept
            return True

    def concepts(
        self, flags: typing.Optional[typing.Set[ConceptFlag]] = None
    ) -> typing.List[Concept]:
        if flags is None:
            return list(self._concepts.values())
        else:
            return [
                concept
                for concept in self._concepts.values()
                if all([flag in concept.flags for flag in flags])
            ]

    def relations(
        self, flags: typing.Optional[typing.Set[RelationFlag]] = None
    ) -> typing.List[Relation]:
        if flags is None:
            return list(self._relations.values())
        else:
            return [
                relation
                for relation in self._relations.values()
                if all([flag in relation.flags for flag in flags])
            ]

    def update(self, message: instrumentation.Message):
        # Send on RelationChangeMessage from subscribed relations to own subscribers
        if isinstance(message, RelationChangeMessage):
            self.notify(message)

    # Some quick helper methods
    def add_prediction_targets(self, uids: typing.List[str]):
        concepts = [Concept(uid, {ConceptFlag.PREDICTION_TARGET}) for uid in uids]
        self.add_concepts(concepts)

    def add_hyponymy_relation(self, sources: typing.List[RelationSource]):
        relation = Relation(
            "chia::Hyponymy",
            sources=sources,
            flags={
                RelationFlag.HYPONYMY,
                RelationFlag.EXPLORE_RIGHT,
                RelationFlag.TRANSITIVE,
            },
        )
        self.add_relation(relation)

    def get_hyponymy_relation(self) -> Relation:
        applicable_relations = self.relations(flags={RelationFlag.HYPONYMY})
        if len(applicable_relations) != 1:
            raise ValueError("Cannot uniquely resolve hyponymy relation")
        else:
            return applicable_relations[0]

    def get_hyponymy_relation_rgraph(self):
        return self.get_hyponymy_relation().rgraph()


class KnowledgeBaseFactory(components.Factory):
    name_to_class_mapping = KnowledgeBase
