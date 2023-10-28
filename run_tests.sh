#!/bin/bash

# if this script fails make sure you've installed everything in requirements-test.txt

python3 -m coverage run -m unittest tests/*.py

python3 -m coverage report -i uavf_2024/*/*.py

rm .coverage