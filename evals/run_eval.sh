#!/bin/bash


#This part is need for OSC users
export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/fs/scratch/xxxx/${USER}/triton_cache
export UV_CACHE_DIR=/fs/scratch/xxxx/${USER}/.cache/uv  #control your uv caches
# Dummy key to prevent import error in safety-eval (WildGuard doesn't actually use it)
export OPENAI_API_KEY="sk-dummy-not-used"
# Disable vLLM V1 multiprocessing so EngineCore runs inline in the spawned subprocess
# rather than forking a grandchild process that loses CUDA visibility on SLURM
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd olmes/oe_eval/dependencies/safety
bash install.sh

cd ../../../

dataset_name=(
    "gsm8k"
    "mbpp"
    "ifeval"
    "harmbench::default"
    "xstest::default"
)
model_path="ajagdale/cse-5525-sft"
for dataset in "${dataset_name[@]}"; do
    echo "Evaluating on ${dataset}..."
    if [ "${dataset}" == "gsm8k" ]; then
        num_shots=8
    else
        num_shots=0
    fi
    uv run olmes \
        --model ${model_path} \
        --task ${dataset} \
        --output-dir $model_path-eval-${dataset} \
        --num-shots ${num_shots}
done