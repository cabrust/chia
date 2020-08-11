import typing

from chia import instrumentation, knowledge
from chia.components import base_models, classifiers


class ModelContainer:
    def __init__(
        self,
        config,
        knowledge_base: knowledge.KnowledgeBase,
        observers: typing.Iterable[instrumentation.Observer] = (),
    ):
        self.knowledge_base = knowledge_base

        self.classifier = classifiers.ClassifierFactory.create(
            config["classifier"], kb=self.knowledge_base, observers=observers
        )

        self.base_model_container = base_models.BaseModelContainerFactory.create(
            config["base_model"], classifier=self.classifier, observers=observers
        )
