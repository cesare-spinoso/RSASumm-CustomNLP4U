#!/bin/bash

cd /home/mila/c/cesare.spinoso/RSASumm/src/rescoring

module load anaconda/3

conda activate rsaumm-novllm

model_versions=(
    "llama3_qfs_standard_beam" 
)

for item in "${model_versions[@]}"
do
    python merge_rescorings.py --config-dir=conf/merge_rescores --config-name=summaries_ans_source_scores +summarizer_name=$item
done
