#!/bin/bash

module load Python/3.10.8-GCCcore-12.2.0

cd temporal-random-walk-test/
source venv/bin/activate

pip install -r requirements.txt

python test.py "$@"
