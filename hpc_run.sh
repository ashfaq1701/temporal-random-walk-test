#!/bin/bash

module load Python/3.10.8-GCCcore-12.2.0

cd temporal-random-walk-test/

rm -rf venv/

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python test.py "$@"

# START=$(date +%s); echo "Start Time: $(date +%T)"; python test.py --use_gpu; END=$(date +%s); echo "End Time: $(date +%T)"; echo "Duration: $(printf '%02d:%02d:%02d\n' $(( (END-START)/3600 )) $(( ((END-START)%3600)/60 )) $(( (END-START)%60 )))"
