import os

import config as pcfg

from chia import containers, helpers, instrumentation


def test_example_experiment():
    """This tests runs the example experiment configuration once."""

    # Set some important environment variables
    os.environ["CHIA_CPU_ONLY"] = "1"  # We need this for cloud testing
    helpers.setup_environment()

    # Read example configuration
    config = pcfg.ConfigurationSet(
        pcfg.config_from_json("examples/configuration.json", read_from_file=True)
    )

    # We need this to log information and to save the results of the experiment
    obs = instrumentation.NamedObservable("Experiment")

    # This IoC container constructs all the necessary objects for our experiment
    experiment_container = containers.ExperimentContainer(config, outer_observable=obs)

    # Run this experiment _without_ exception handling
    experiment_container.runner.run()

    # Make sure all the data is saved
    obs.send_shutdown(successful=True)
