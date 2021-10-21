import os

import config as pcfg
import pytest

from chia import containers, helpers, instrumentation
from chia.components import classifiers


@pytest.mark.parametrize(
    "hc_name", classifiers.ClassifierFactory.name_to_class_mapping.keys()
)
@pytest.mark.parametrize("held_out_test_set", (False, True))
@pytest.mark.parametrize("validate_on_test_set", (False, True))
def test_example_experiment(
    hc_name: str, held_out_test_set: bool, validate_on_test_set: bool
):
    """This tests runs the example experiment configuration once."""

    # Set some important environment variables
    os.environ["CHIA_CPU_ONLY"] = "1"  # We need this for cloud testing
    helpers.setup_environment()

    # Read example configuration
    config_save = pcfg.ConfigurationSet(
        pcfg.config_from_dict({"model.classifier.name": hc_name}),
        pcfg.config_from_dict({"runner.held_out_test_set": held_out_test_set}),
        pcfg.config_from_dict({"runner.validate_on_test_set": validate_on_test_set}),
        pcfg.config_from_json("examples/configuration.json", read_from_file=True),
    )
    config_save["runner.save_path"] = "test_epoch_runner_load_save_data"

    # We need this to log information and to save the results of the experiment
    obs_save = instrumentation.NamedObservable("ExperimentSave")

    # This IoC container constructs all the necessary objects for our experiment
    experiment_container_save = containers.ExperimentContainer(
        config_save, outer_observable=obs_save
    )

    # Get some sample data
    if experiment_container_save.runner.max_test_samples is not None:
        test_samples = experiment_container_save.dataset.test_pool(0, "label_gt")[
            : experiment_container_save.runner.max_test_samples
        ]
    else:
        test_samples = experiment_container_save.dataset.test_pool(0, "label_gt")

    # Run this experiment _without_ exception handling
    experiment_container_save.runner.run()

    # Get predictions
    mdl_saved = (
        experiment_container_save.model_container.base_model_container.base_model
    )
    pred_saved = mdl_saved.predict_probabilities(test_samples, "label_pred_dist")

    # Make sure all the data is saved
    obs_save.send_shutdown(successful=True)

    #
    # Now comes the loading part...
    #

    # Read example configuration
    config_load = pcfg.ConfigurationSet(
        pcfg.config_from_dict({"model.classifier.name": hc_name}),
        pcfg.config_from_json("examples/configuration.json", read_from_file=True),
    )
    config_load["runner.load_path"] = "test_epoch_runner_load_save_data"
    config_load["runner.epochs"] = 0

    # We need this to log information and to save the results of the experiment
    obs_load = instrumentation.NamedObservable("ExperimentLoad")

    # This IoC container constructs all the necessary objects for our experiment
    experiment_container_load = containers.ExperimentContainer(
        config_load, outer_observable=obs_load
    )

    # Run this experiment _without_ exception handling
    experiment_container_load.runner.run()

    mdl_loaded = (
        experiment_container_load.model_container.base_model_container.base_model
    )
    pred_loaded = mdl_loaded.predict_probabilities(test_samples, "label_pred_dist")

    for sample_loaded, sample_saved in zip(pred_loaded, pred_saved):
        dist_loaded = sample_loaded.get_resource("label_pred_dist")
        dist_saved = sample_saved.get_resource("label_pred_dist")
        assert dist_loaded == dist_saved

    # Make sure all the data is saved
    obs_load.send_shutdown(successful=True)
