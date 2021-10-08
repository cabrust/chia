import os

import config as pcfg
import networkx as nx
import pytest

from chia import containers, helpers, instrumentation, knowledge
from chia.components import extrapolators
from chia.knowledge import information_content


@pytest.mark.parametrize(
    "extrapolator_name", extrapolators.ExtrapolatorFactory.name_to_class_mapping.keys()
)
@pytest.mark.parametrize(
    "ic_method_name",
    information_content.InformationContentCalculatorFactory.name_to_class_mapping.keys(),
)
def test_extrapolators(extrapolator_name: str, ic_method_name: str):
    """This tests runs the example experiment configuration once."""

    # Set some important environment variables
    os.environ["CHIA_CPU_ONLY"] = "1"  # We need this for cloud testing
    helpers.setup_environment()

    # Read example configuration
    config = pcfg.ConfigurationSet(
        pcfg.config_from_dict({"model.classifier.name": "chillax"}),
        pcfg.config_from_dict({"extrapolator.name": extrapolator_name}),
        pcfg.config_from_dict({"extrapolator.ic_method": ic_method_name}),
        pcfg.config_from_dict({"extrapolator.apply_ground_truth": True}),
        pcfg.config_from_json("examples/configuration.json", read_from_file=True),
    )

    # We need this to log information and to save the results of the experiment
    obs = instrumentation.NamedObservable("Experiment")

    # This IoC container constructs all the necessary objects for our experiment
    experiment_container = containers.ExperimentContainer(config, outer_observable=obs)

    # Get important objects from the container
    knowledge_base: knowledge.KnowledgeBase = experiment_container.knowledge_base
    extrapolator: extrapolators.Extrapolator = experiment_container.extrapolator
    rgraph: nx.DiGraph = knowledge_base.get_hyponymy_relation_rgraph()

    # Get all concepts
    concept_uids = [concept.uid for concept in knowledge_base.concepts()]
    imprecise_concept_uids = [
        concept_uid
        for concept_uid in concept_uids
        if rgraph.out_degree[concept_uid] > 0
    ]

    for imprecise_concept_uid in imprecise_concept_uids:
        # Set 1: All concepts that apply to imprecise_concept_uid
        set_1 = set(nx.ancestors(rgraph, imprecise_concept_uid)).union(
            set([imprecise_concept_uid])
        )

        # Set 2: All concepts that imprecise_concept_uid applies to
        set_2 = set(nx.descendants(rgraph, imprecise_concept_uid)).union(
            set([imprecise_concept_uid])
        )

        # Generate a somewhat realistic input
        probabilities = {}
        for concept_uid in concept_uids:
            if concept_uid in set_1:
                probabilities[concept_uid] = 1.0
            elif concept_uid in set_2:
                probabilities[concept_uid] = 0.2
            else:
                probabilities[concept_uid] = 0.0

        result = extrapolator.extrapolate([(imprecise_concept_uid, probabilities)])[0]

        # Since apply_ground_truth is true, this should always work. Either way, the prediction should never "disagree" with the ground truth
        assert result in set_2, f"While testing for gt={imprecise_concept_uid}"

    # Make sure all the data is saved
    obs.send_shutdown(successful=True)
