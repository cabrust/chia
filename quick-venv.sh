#!/bin/bash
python3.9 -m venv venv
source venv/bin/activate
python3.9 -m pip install -U pip setuptools
python3.9 setup.py develop
