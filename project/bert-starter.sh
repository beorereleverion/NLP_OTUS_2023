#!/bin/bash

model_names=("ruBert_base" "ruBert_large")
thresholds=("50" "100")
lr_values=("2e-4" "2e-5")
eps_values=("1e-7" "1e-8")
sentence_lengths=("128" "256")
batch_sizes=("16" "32")

for model_name in "${model_names[@]}"; do
  for threshold in "${thresholds[@]}"; do
    for lr_value in "${lr_values[@]}"; do
      for eps_value in "${eps_values[@]}"; do
        for sentence_length in "${sentence_lengths[@]}"; do
          for batch_size in "${batch_sizes[@]}"; do

            export MODEL_NAME="$model_name"
            export TRESHOLD="$threshold"
            export LR_VALUE="$lr_value"
            export EPS_VALUE="$eps_value"
            export SENTENCE_LENGTH="$sentence_length"
            export BATCH_SIZE="$batch_size"
            
            python3.10 project.py || exit
          done
        done
      done
    done
  done
done
