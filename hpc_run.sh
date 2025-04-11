#!/bin/bash

module load Python/3.10.8-GCCcore-12.2.0

cd temporal-random-walk-test/

rm -rf venv/

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python test.py "$@" > output.txt 2>&1
