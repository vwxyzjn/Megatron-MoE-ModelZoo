#!/bin/bash
set -euxo pipefail

# Path to Megatron-MoE-Scripts
WORKSPACE=$(dirname "$(readlink -f "$0")")

# Benchmarking configurations (must be set)
export MODEL=${MODEL:-"your_own_model"}
export CLUSTER=${CLUSTER:-"your_own_cluster"}
export MCORE_RELEASE_VERSION=${MCORE_RELEASE_VERSION:-"your_own_megatron_version"} # Version and release info
export MEGATRON_PATH=${MEGATRON_PATH:-"your_own_megatron_path"} # Path to Megatron-LM
export WANDB_API_KEY=${WANDB_API_KEY:-"your_own_wandb_api_key"} # Wandb API key

# Load common configurations
source "${WORKSPACE}/runtime_configs/benchmarking/common.conf"
# Load model-specific configurations
source "${WORKSPACE}/runtime_configs/benchmarking/runtime.conf"
# Load cluster configurations
source "${WORKSPACE}/cluster_configs/benchmarking/${CLUSTER}.conf"

# Initialize training parameters
TRAINING_PARAMS=${TRAINING_PARAMS:-""}

# Process training parameters
if [[ -f ${TRAINING_PARAMS_PATH} ]]; then
    envsubst < ${TRAINING_PARAMS_PATH} > ${TRAINING_PARAMS_PATH}.tmp
    TRAINING_PARAMS_PATH=${TRAINING_PARAMS_PATH}.tmp
else
    echo "Error: TRAINING_PARAMS_PATH does not exist: ${TRAINING_PARAMS_PATH}."
    exit 1
fi

# Extract training parameters to export
TRAINING_PARAMS_FROM_CONFIG=$(yq '... comments="" | .MODEL_ARGS | to_entries | .[] | 
    select(.value != "false") | 
    with(select(.value == "true"); .value = "") | 
    with(select(.key == "--pipeline-model-parallel-layout"); .value = (.value | @json)) | 
    [.key + " " + .value] | join("")' ${TRAINING_PARAMS_PATH} | tr '\n' ' ')
TRAINING_PARAMS="${TRAINING_PARAMS} ${TRAINING_PARAMS_FROM_CONFIG}"

# Append any command line arguments to TRAINING_PARAMS
if [[ $# -gt 0 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} $@"
fi

# Extract environment variables to export
ENV_VARS=$(yq '... comments="" | .ENV_VARS | to_entries | .[] | [.key + "=" + .value] | join(" ")' ${TRAINING_PARAMS_PATH})
while IFS='=' read -r KEY VALUE; do
    if [[ -n ${KEY} ]]; then
        export "${KEY}"="${VALUE}"
        echo "${KEY}=${VALUE}"
    fi
done < <(echo "${ENV_VARS}" | tr ' ' '\n')

# Virtual pipeline parallelism arguments
if [[ ${VPP} -gt 1 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --num-layers-per-virtual-pipeline-stage ${LAYERS_PER_VP}"
fi

# Uneven pipeline parallelism arguments
if [[ $((NUM_LAYERS % PP)) -ne 0 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --decoder-first-pipeline-num-layers ${PP_FIRST} --decoder-last-pipeline-num-layers ${PP_LAST}"
fi

# FP8 arguments
if [[ ${PR} == "fp8" ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
fi

# Profile command
if [[ ${PROFILE} -eq 1 ]]; then
    NSYS_PATH="${OUTPUT_PATH}/nsys"
    DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
    mkdir -p "${NSYS_PATH}"
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-memory-usage true \
        -f true -x true \
        -o ${NSYS_PATH}/${MODEL}-benchmarking-${DATETIME}"
    TRAINING_PARAMS="${TRAINING_PARAMS} --profile --profile-step-start 50 --profile-step-end 55 --profile-ranks 0 "
else
    PROFILE_CMD=""
fi

# Distributed training settings
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
DISTRIBUTED_ARGS=(
    --nproc_per_node ${GPUS_PER_NODE}
    --nnodes ${NNODES}
    --master_addr ${MASTER_ADDR}
    --master_port ${MASTER_PORT}
)

# Start training
cd ${MEGATRON_PATH} || {
    echo "Error: Failed to change directory to ${MEGATRON_PATH}"
    exit 1
}
${PROFILE_CMD} torchrun ${DISTRIBUTED_ARGS[@]} ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}
