#!/bin/bash

module load Python/3.10.8-GCCcore-12.2.0

cd temporal-random-walk-test/

# Generate a random venv name with timestamp
VENV_NAME="venv_$(date +%Y%m%d_%H%M%S)_$$"

python -m venv $VENV_NAME
source $VENV_NAME/bin/activate

pip install -r requirements.txt

python test.py "$@"

# Cleanup: deactivate and remove the venv
deactivate
rm -rf $VENV_NAME
