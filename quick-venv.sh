#!/bin/bash
python3.8 -m venv venv
source venv/bin/activate
python3.8 -m pip install -U pip setuptools
python3.8 setup.py develop
