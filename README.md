# CHIA: Concept Hierarchies for Incremental and Active Learning
![PyPI](https://img.shields.io/pypi/v/chia)
![PyPI - License](https://img.shields.io/pypi/l/chia)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/chia)
![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability/cvjena/chia)
![codecov](https://codecov.io/gh/cvjena/chia/branch/main/graph/badge.svg)

CHIA is a collection of methods and helper functions centered around hierarchical classification in a lifelong learning environment.
It forms the basis for some of the experiments and tools developed at [Computer Vision Group Jena](http://www.inf-cv.uni-jena.de/).

## Methods
CHIA implements:
 * **One-Hot Classifier** as a baseline.
 * **Probabilistic Hierarchical Classifier** Brust, C. A., & Denzler, J. (2019, November). *Integrating domain knowledge: using hierarchies to improve deep classifiers*. In Asian Conference on Pattern Recognition (ACPR) (pp. 3-16). Springer, Cham.
 * **CHILLAX** Brust, C. A., Barz, B., & Denzler, J. (2021, January). *Making Every Label Count: Handling Semantic Imprecision by Integrating Domain Knowledge*. In 2020 25th International Conference on Pattern Recognition (ICPR) (pp. 6866-6873). IEEE.
 * **Self-Supervised CHILLAX** Brust, C. A., Barz, B., & Denzler, J. (2021, April). *Self-Supervised Learning from Semantically Imprecise Data*. arXiv preprint arXiv:2104.10901.
 * **Semantic Label Sharing** Fergus, R., Bernal, H., Weiss, Y., & Torralba, A. (2010, September). *Semantic label sharing for learning with many categories*. In European Conference on Computer Vision (pp. 762-775). Springer, Berlin, Heidelberg.

## Datasets
The following datasets are integrated into CHIA:
 * CORe50
 * CUB200-2011
 * (i)CIFAR-100
 * ImageNet ILSVRC2012
 * NABirds

## Requirements
CHIA depends on:
* python-configuration == 0.7.1
* nltk ~= 3.5
* imageio ~= 2.6
* pillow ~= 8.0
* gputil ~= 1.4.0
* networkx ~= 2.4
* numpy ~= 1.19.2
* tensorflow-addons == 0.14.0
* tensorflow == 2.6.0

## Installation
To install, simply run:
```bash
pip install chia
```
or clone this repository, and run:
```bash
pip install -U pip setuptools
python setup.py develop
```

We also include the shell script `quick-venv.sh`, which creates a virtual environment and install CHIA for you.

## Getting Started
To run the [example experiment](examples/experiment.py) which makes sure that everything works, use the following command:
```bash
python examples/experiment.py examples/configuration.json
```
After a few minutes, the last lines of output should look like this:
```text
[DEBUG] [ExceptionShroud]: Leaving exception shroud without exception
[SHUTDOWN] [Experiment] Successful: True
```

## Citation
If you use CHIA for your research, kindly cite:
> Brust, C. A., & Denzler, J. (2019, November). Integrating domain knowledge: using hierarchies to improve deep classifiers. In Asian Conference on Pattern Recognition (pp. 3-16). Springer, Cham.

You can refer to the following BibTeX:
```bibtex
@inproceedings{Brust2019IDK,
author = {Clemens-Alexander Brust and Joachim Denzler},
booktitle = {Asian Conference on Pattern Recognition (ACPR)},
title = {Integrating Domain Knowledge: Using Hierarchies to Improve Deep Classifiers},
year = {2019},
}
```
