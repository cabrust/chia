from chia import instrumentation
from chia.knowledge.concept import Concept, ConceptFlag
from chia.knowledge.knowledge_base import KnowledgeBase
from chia.knowledge.messages import ConceptChangeMessage, RelationChangeMessage
from chia.knowledge.relation import Relation, RelationFlag


class Discovery(instrumentation.Observer):
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def update(self, message: instrumentation.Message):
        if isinstance(message, ConceptChangeMessage) or isinstance(
            message, RelationChangeMessage
        ):
            for relation in self.knowledge_base.relations():
                self.update_relation(relation)

    def update_relation(self, relation: Relation):
        if self._is_compatible(relation):

            discovered_pairs = set()
            discovered_concepts = set()
            known_concepts = set()
            for concept in self.knowledge_base.concepts():
                discovered_concepts |= {concept.uid}
                known_concepts |= {concept.uid}

                # Explore to the right
                uid_left = concept.uid

                for relation_source in relation.sources:
                    uids_to_right = relation_source.get_right_for(uid_left)
                    discovered_pairs |= {
                        (uid_left, uid_right) for uid_right in uids_to_right
                    }
                    discovered_concepts |= set(uids_to_right)

            # Add relation pairs first. This will trigger another discovery.
            relation.add_pairs(discovered_pairs)

            # Then, add new concepts. This will happen in the second run. The outer run does
            # it again, but it will not have an effect.
            discovered_concepts -= known_concepts
            concepts_for_kb = [
                Concept(uid, {ConceptFlag.AUTO_DISCOVERED})
                for uid in discovered_concepts
            ]
            self.knowledge_base.add_concepts(concepts_for_kb)

        else:
            # Skip relation
            pass

    @staticmethod
    def _is_compatible(relation: Relation) -> bool:
        return (
            RelationFlag.SYMMETRIC not in relation.flags
            and RelationFlag.REFLEXIVE not in relation.flags
            and RelationFlag.TRANSITIVE in relation.flags
            and RelationFlag.EXPLORE_RIGHT in relation.flags
        )
