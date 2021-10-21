# Architecture of CHIA

CHIA is designed with configurability and extensibility in mind. It relies on the excellent [python-configuration] library to parse configuration files. Almost all classes are instantiated using parameters from such configuration files. Because we apply IoC wherever possible, most implementations of methods could feasibly be used outside of CHIA.

The bulk of CHIA classes fall into one of the following categories:
 - [*Containers*](#containers), which instantiate and combine components, e.g., to build an experimental environment.
 - [*Components*](#components), which actually implement things, e.g., classifiers, datasets etc.
 - *Instrumentation*, which performs logging and collection of metrics/results.
 - *Knowledge*, which contains classes that manage concept hierarchies, relations etc.

## Containers
With containers, we combine components that have mutual dependencies in a transparent fashion, such that they are still instantiated using a user-provided configuration. The main container in CHIA is the experiment container.

### Experiment Container
The class [`ExperimentContainer`] collects all of the different components and other classes that make up an experiment, including:
 - Information about the experiment (metdata).
 - Information about the environment (GPUs, environment variables...)
 - Observers for logging and metric collection.
 - Structured exception handling using the aforementioned observers.
 - A knowledge base, which contains concepts/classes and relations/hierarchies.
 - A [dataset](#dataset).
 - An [interactor](#interactor).
 - [Sample transformers](#sample-transformer).
 - A model container, consisting of a [base model](#base-model) and a [classifier](#classifier).
 - [Evaluators](#evaluator).
 - An [extrapolator](#extrapolator).
 - A [base model](#base-model).
 - A [runner](#runner).

This way, a single JSON file can be used to completely specify an experiment in most cases.

## Components

### Base Model

A base model implements methods to make predictions and observe training examples. It can save and restore its state from the file system. Currently, there is only one implementation ([`KerasBaseModel`]). New implementations should inherit from the base class [`ProbabilityOutputModel`], or if predictions are purely discrete, [`IncrementalModel`].

The main implementation [`KerasBaseModel`] has the following exchangable sub-components:
 - Feature Extractor: Extracts features from images, i.e. the "backbone".
 - Preprocessor: Converts images to floating point representation, performs normalizations etc.
 - Data Augmentation: Random flips, rotations, crops, scales etc.
 - Trainer: Holds optimization algorithm and state, learning rate schedules etc.

The base model also holds a reference to a [classifier](#classifier), i.e. the "last layer" of a neural network.

### Classifier
A classifier processes features and returns a probability for each prediction target (see [dataset](#dataset)). The following classifiers are currently implemented:

 - One-hot encoding with softmax.
 - Hierarchical classifier (Brust '18 ACPR).
 - CHILLAX (Brust '21 ICPR).
 - Label sharing (Fergus '10 ECCV).

New implementations should inherit from [`KerasHierarchicalClassifier`] unless a new base model is also added.

### Dataset

A dataset implements methods to retrieve instances of [`Sample`], which are grouped into arbitrarily main train and test pools.

Each dataset should define a set of *prediction targets*, which are the classes/concepts that are possible outputs of the classifier. Typically, these would be the leaf nodes of a hierarchy. The dataset should also include a *hyponymy relation source*, which is queried to build the concept hierarchy. By default, CHIA also queries WordNet.

Furthermore, each dataset should specify a *namespace* such that class/concept names that are unique to the dataset can be recognized as such and distinguished from classes/concepts from other sources. For example, the CORe50 dataset might have a class "plug_adapter", which is uniquely identified by `CORe50::plug_adapter`. The hyponymy relation includes an element `CORe50::plug_adapter` --> `WordNet3.0::plug.n.05`. Using these prefixes, it is clear where each concept comes from.

The following datasets are implemented "natively" in CHIA:

 * CORe50
 * CUB200-2011
 * (i)CIFAR-100
 * iCubWorld28
 * ImageNet ILSVRC2012
 * iNaturalist 2018
 * NABirds

New implementations should inherit from the base class [`Dataset`]. However, it is not necessary to add code in order to use a new dataset. Instead, [`JSONDataset`] should be used (see [Using your own dataset])

### Evaluator

Evaluators compare predictions with ground truth on a set of samples and return some fitness measure. Because testing can be done in batches, the evaluators calculate the measures incrementally each time a new set of samples is presented, and have to be reset after testing.

New implementations should inherit from the base class [`Evaluator`]. The following evaluators are implemented:

 - Accuracy,
 - top-k accuracy,
 - hierarchical (hF1, hPRE, hREC, semantic distance).

### Extrapolator

Extrapolators increase the precision of imprecise labels during training. The default is to do no extrapolation, i.e. leave the training data as is. Extrapolators are currently only used for the experiments in [self-supervised CHILLAX].

### Interactor

Interactors are used to simulator or actually perform user interaction to acquire labels. For typical experiments, labels are already available, and one of the two interactors should be used:

 - *Oracle*, which simply uses the ground truth from the dataset as labels. Use this.
 - *NoisyOracle*, which uses the ground truth from the dataset, but reduces accuracy and precision to simulate inexperienced annotators.

New implementations should inherit from [`Interactor`].

### Runner

Runners contain all experimental logic. The default implementation is [`EpochRunner`], which load one training and test pool each, runs a fixed number of training epochs and performs evaluations on the validation data in between. It can also save and restore model states.

New implementations should inherit from [`Runner`], for example, to implement complex incremental learning scenarios.

*Note:* CHIA differentiates between training, validation and test data. In practice, make sure to only use an actual held-out test set once. By default, the runner will not use a held-out test set. The implemented datasets return validation and test sets as specified by the respective authors. *Don't optimize hyperparameters on your test set! If a dataset does not come with a validation split, please make your own instead of using the test set more than once.*

### Sample Transformer

Sample transformers are similar to interactors in that they affect the data, but more powerful. A sample transformer can change both training and test data completely. Currently, there is only one implementation, [`IdentitySampleTransformer`], which does nothing. However, these could be used for any number of experiments, including adding noise to images, adding label noise to the test data etc.

New implementations should inherit from [`SampleTransformer`].


[Using your own dataset]: dataset.md
[python-configuration]: https://github.com/tr11/python-configuration
[`Sample`]: /chia/data/__init__.py
[`Dataset`]: /chia/components/datasets/dataset.py
[`JSONDataset`]: /chia/components/datasets/json_dataset.py
[`KerasBaseModel`]: /chia/components/base_models/keras/keras_basemodel.py
[`ProbabilityOutputModel`]: /chia/components/base_models/incremental_model.py
[`IncrementalModel`]: /chia/components/base_models/incremental_model.py
[`Interactor`]: /chia/components/interactors/interactor.py
[`EpochRunner`]: /chia/components/runners/epoch.py
[`Runner`]: /chia/components/runners/runner.py
[`IdentitySampleTransformer`]: /chia/components/transformers/identity.py
[`SampleTransformer`]: /chia/components/transformers/sample_transformer.py
[self-supervised CHILLAX]: https://arxiv.org/abs/2104.10901
[`ExperimentContainer`]: /chia/containers/experiment.py
[`KerasHierarchicalClassifier`]: /chia/components/keras_hierarchicalclassification.py