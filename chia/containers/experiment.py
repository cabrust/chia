import random
import typing

from chia import helpers, instrumentation, knowledge
from chia.components import datasets, evaluators, interactors, runners
from chia.components.base_models import incremental_model
from chia.components.datasets.dataset import Dataset
from chia.components.interactors import interactor
from chia.containers import model


class ExperimentMetadata(typing.NamedTuple):
    name: str
    run_id: str


class ExperimentContainer:
    def __init__(self, config, outer_observable: instrumentation.Observable):
        # Configuration
        self.config = config

        # Experiment metadata
        self.experiment_metadata = ExperimentMetadata(
            name=config["meta"]["name"], run_id=("%08x" % random.randrange(2 ** 32))
        )

        # Environment info
        self.environment_info = helpers.EnvironmentInfo()

        # Load all observers
        self.observers: typing.List[instrumentation.Observer] = [
            instrumentation.ObserverFactory.create(
                sub_config,
                experiment_metadata=self.experiment_metadata,
                environment_info=self.environment_info,
            )
            for sub_config in config["observers"]
        ]
        for observer in self.observers:
            outer_observable.register(observer)

        # Make an exception shroud
        self.exception_shroud = instrumentation.ExceptionShroudFactory.create(
            dict(), observers=self.observers
        )

        with self.exception_shroud:
            # Build knowledge base
            self.knowledge_base = knowledge.KnowledgeBaseFactory.create(
                dict(), observers=self.observers
            )

            # Dataset
            self.dataset: Dataset = datasets.DatasetFactory.create(
                config["dataset"], observers=self.observers
            )

            # Interactor
            self.interactor: interactor.Interactor = (
                interactors.InteractorFactory.create(
                    config["interactor"],
                    kb=self.knowledge_base,
                    observers=self.observers,
                )
            )

            # Model container
            self.model_container = model.ModelContainer(
                config["model"],
                knowledge_base=self.knowledge_base,
                observers=self.observers,
            )

            # Instantiate all evaluators
            self.evaluators: typing.List[evaluators.Evaluator] = [
                evaluators.EvaluatorFactory.create(
                    sub_config, kb=self.knowledge_base, observers=self.observers
                )
                for sub_config in config["evaluators"]
            ]

            # Allow outside access
            self.classifier = self.model_container.classifier
            self.base_model_container = self.model_container.base_model_container
            self.base_model: incremental_model.IncrementalModel = (
                self.model_container.base_model_container.base_model
            )

            # Run KB setup
            self._setup_knowledge_base()

            # Build runner
            self.runner: runners.Runner = runners.RunnerFactory.create(
                config["runner"], experiment_container=self, observers=self.observers
            )

    def _setup_knowledge_base(self):
        """Sets up the prediction targets and hyponymy relation from the dataset."""
        # Get prediction targets
        self.knowledge_base.add_prediction_targets(self.dataset.prediction_targets())

        # Add relation sources
        relation_sources = [self.dataset.get_hyponymy_relation_source()]
        if self.config["with_wordnet"]:
            relation_sources += [knowledge.WordNetAccess()]
        self.knowledge_base.add_hyponymy_relation(relation_sources)
