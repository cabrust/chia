import argparse

import config as pcfg

from chia import containers, helpers, instrumentation


def main(config_files):
    # Set up buffered observer
    buffered_observer = instrumentation.ObserverFactory.create({"name": "buffered"})

    # Set some important environment variables and validate the GPU configuration
    helpers.setup_environment([buffered_observer])

    # Read configuration files from command line arguments
    configs = [
        pcfg.config_from_json(config_file, read_from_file=True)
        for config_file in config_files
    ]

    # Read user configuration file ~/.chiarc
    # This file contains user-specific paths for datasets
    configs += [helpers.get_user_config()]
    config = pcfg.ConfigurationSet(*configs)

    # We need this to log information and to save the results of the experiment
    obs = instrumentation.NamedObservable("Experiment")

    # This IoC container constructs all the necessary objects for our experiment
    experiment_container = containers.ExperimentContainer(config, outer_observable=obs)

    # Replay the buffer
    buffered_observer.replay_messages(obs)

    # Run this experiment with exception handling
    with experiment_container.exception_shroud:
        experiment_container.runner.run()

    # Make sure all the data is saved
    obs.send_shutdown(successful=True)


if __name__ == "__main__":
    # We want to supply configuration files over the command line
    parser = argparse.ArgumentParser(prog="experiment")
    parser.add_argument("config_file", type=str, nargs="+")
    args = parser.parse_args()
    main(config_files=args.config_file)
