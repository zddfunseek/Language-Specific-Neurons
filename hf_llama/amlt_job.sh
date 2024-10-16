#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CODE_DIR=../code
TASK=$1
MODEL_PATH=$2
MODEL_NAME="${MODEL_PATH##*/}"
OUTPUT_PATH=/mnt/output/OurExperiments
BENCHMARK_DIR=/mnt/input/OurExperiments/glan_eval_benchmarks/eval/
PROMPT_TEMPLATE=none  # options: [none, alpaca, alpaca_force_ans, alpaca_cot, chat_format]
TASK_TYPE=none # options: [none, general_mc, general, multi_choice, run_prob, code]
N_SHOT=0

cd $CODE_DIR

echo "Evaluating ${MODEL_NAME} Model..."
echo "Evaluating ${TASK} dataset..."
ls -l

python -m main \
    --tasks $TASK \
    --model_name_or_path ${MODEL_PATH} \
    --data_dir $BENCHMARK_DIR/${TASK} \
    --save_dir $OUTPUT_PATH/${TASK}/${MODEL_NAME} \
    --prompt_template ${PROMPT_TEMPLATE} \
    --n_shot ${N_SHOT} \
    --tasktype ${TASK_TYPE}

echo 'finish evaluated'
