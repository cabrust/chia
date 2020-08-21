# CHIA: Concept Hierarchies for Incremental and Active Learning
![PyPI](https://img.shields.io/pypi/v/chia)
![PyPI - License](https://img.shields.io/pypi/l/chia)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/chia)
![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability/cvjena/chia)
![codecov](https://codecov.io/gh/cvjena/chia/branch/main/graph/badge.svg)

CHIA is a collection of methods and helper functions centered around hierarchical classification in a lifelong learning environment.
It forms the basis for some of the experiments and tools developed at [Computer Vision Group Jena](http://www.inf-cv.uni-jena.de/).

## Requirements
CHIA depends on:
* python-configuration ~= 0.7
* nltk ~= 3.5
* imageio ~= 2.6
* pillow ~= 7.1.0
* gputil ~= 1.4.0
* networkx ~= 2.4
* numpy ~= 1.18.5
* tensorflow-addons == 0.11.1
* tensorflow == 2.3.0

Optional dependencies:

* tables ~= 3.6.1
* pandas ~= 1.0.4
* sacred ~= 0.8.1
* pyqt5 ~= 5.15.0
* scikit-image ~= 0.17.2
* scikit-learn ~= 0.23.1
* scipy == 1.4.1
* matplotlib ~= 3.2.1

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