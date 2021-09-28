# CHIA: Concept Hierarchies for Incremental and Active Learning
![PyPI](https://img.shields.io/pypi/v/chia)
![PyPI - License](https://img.shields.io/pypi/l/chia)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/chia)
![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability/cvjena/chia)
![codecov](https://codecov.io/gh/cvjena/chia/branch/main/graph/badge.svg)

CHIA implements methods centered around hierarchical classification in a lifelong learning environment.
It forms the basis for some of the experiments and tools developed at [Computer Vision Group Jena](http://www.inf-cv.uni-jena.de/).

**Methods**\
CHIA implements:
 * **One-Hot Softmax Classifier** as a baseline.
 * **Probabilistic Hierarchical Classifier** Brust, C. A., & Denzler, J. (2019). *Integrating domain knowledge: using hierarchies to improve deep classifiers*. In Asian Conference on Pattern Recognition (ACPR)
 * **CHILLAX** Brust, C. A., Barz, B., & Denzler, J. (2021). *Making Every Label Count: Handling Semantic Imprecision by Integrating Domain Knowledge*. In International Conference on Pattern Recognition (ICPR).
 * **Self-Supervised CHILLAX** Brust, C. A., Barz, B., & Denzler, J. (2021). *Self-Supervised Learning from Semantically Imprecise Data*. arXiv preprint arXiv:2104.10901.
 * **Semantic Label Sharing** Fergus, R., Bernal, H., Weiss, Y., & Torralba, A. (2010). *Semantic label sharing for learning with many categories*. In European Conference on Computer Vision (ECCV).

**Datasets**\
CHIA has integrated support including hierarchies for a number of popular datasets. See [here](docs/architecture.md#dataset) for a complete list.


## Installation and Getting Started
CHIA is available on PyPI. To install, simply run:
```bash
pip install chia
```
or clone this repository, and run:
```bash
python setup.py develop
```

To run the [example experiment](examples/experiment.py) which makes sure that everything works, use the following command:
```bash
python examples/experiment.py examples/configuration.json
```
After a few minutes, the last lines of output should look like this:
```text
[SHUTDOWN] [Experiment] Successful: True
```

## Documentation
The following articles explain more about CHIA:
 * [Architecture](docs/architecture.md) explains the overall construction. It also includes reference descriptions of most classes.
 * [Configuration](docs/configuration.md) describes how experiments and CHIA itself are configured.

## Citation
If you use CHIA for your research, kindly cite:
> Brust, C. A., & Denzler, J. (2019). Integrating domain knowledge: using hierarchies to improve deep classifiers. In Asian Conference on Pattern Recognition. Springer, Cham.

You can refer to the following BibTeX:
```bibtex
@inproceedings{Brust2019IDK,
author = {Clemens-Alexander Brust and Joachim Denzler},
booktitle = {Asian Conference on Pattern Recognition (ACPR)},
title = {Integrating Domain Knowledge: Using Hierarchies to Improve Deep Classifiers},
year = {2019},
doi = {10.1007/978-3-030-41404-7_1}
}
```
