# Configuration

A typical CHIA experiment collects its configuration from two main sources:
 - One or more [JSON files](#json-file-configuration) supplied via the command line. These files contain experiment-specific settings.
 - The [user configuration](#user-configuration) in `~/.chiarc`. This should be used for machine- or user-specific settings, e.g., the location of a dataset.

All configuration sources are merged into a single [python-configuration] object as illustrated by the [example experiment]:

```python
# Read configuration files from command line arguments
configs = [
    pcfg.config_from_json(config_file, read_from_file=True)
    for config_file in config_files
]

# Read user configuration file ~/.chiarc
# This file contains user-specific paths for datasets
configs += [helpers.get_user_config()]
config = pcfg.ConfigurationSet(*configs)
```

This dictionary is then used to instantiate an [experiment container]:
```python
experiment_container = containers.ExperimentContainer(config, ...)
```

## JSON file configuration
Each element of the [experiment container] is assigned a key, where the mapping is specified in the constructor of [`ExperimentContainer`]. A basic skeleton of a configuration file looks as follows:
```json
{
    "meta": {...},
    "evaluators": [...],
    "with_wordnet": true,
    "interactor": {...},
    "observers": [...],
    "sample_transformers": [...],
    "runner": {...},
    "dataset": {...},
    "model": {...}
}
```

A full working example configuration is available [here](/examples/configuration.json). Most values are dictionaries (or arrays of dictionaries), which are used to initialize objects. The value assigned to `name` specifies the desired class. For example, the following snippet describes the construction of a classifier:
```json
{
    "name": "keras_onehot",
    "l2": 1e-5,
}
```
When passed to [`ClassifierFactory`], it will result in an object of type [`OneHotEmbeddingBasedKerasHC`]. The constructor parameter `l2` is set to 1e-5, overriding the default value of 5e-5. If keys do not match constructor parameters, a warning is issued. If a constructor parameter does not have a default value and the respective key is missing from the dictionary, a fatal error occurs.

Some configuration entries can seem overly verbose, e.g.:
```json
{
    "interactor": {
        "name": "oracle"
    }
}
```
If you do not want to pass any constructor parameters, you can replace the dictionary with a single string indicating the desired class:
```json
{
    "interactor": "oracle"
}
```

## User configuration
Some values are not specific to an experiment, but rather to the environment. It is suggested to keep these in a separate file in the home directory as `~/.chiarc`. For example, consider the following snippet:

```json
{
    "dataset.nabirds_userdefaults.base_path": "/home/brust/datasets/nabirds"
}
```

The suffix `_userdefaults` tells `Factory` that when a dataset with name `nabirds` is constructed and no value for `base_path` is provided otherwise, to use the supplied path.

[example experiment]: /examples/experiment.py
[experiment container]: architecture.md#experiment-container
[python-configuration]: https://github.com/tr11/python-configuration
[`ExperimentContainer`]: /chia/containers/experiment.py
[`ClassifierFactory`]: /chia/components/classifiers/__init__.py
[`OneHotEmbeddingBasedKerasHC`]: /chia/components/classifiers/keras_onehot_hc.py