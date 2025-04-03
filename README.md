# Megatron MoE Testing Guide

Built on the powerful Megatron-Core framework, this guide delivers detailed instructions and best practices for testing cutting-edge Mixtral, DeepSeek, and Qwen series models. By following this guide, users can ensure optimal performance and reliability, unlocking the full potential of these innovative models.

# Table of Contents
- [Megatron MoE Testing Guide](#megatron-moe-testing-guide)
- [Table of Contents](#table-of-contents)
- [0. Container Setup](#0-container-setup)
- [1. Environment Setup](#1-environment-setup)
  - [1.1. Login Node Setup](#11-login-node-setup)
- [2. Performance Benchmarking](#2-performance-benchmarking)
  - [2.1. Test Script Setup](#21-test-script-setup)
  - [2.2. Runtime Configuration Setup](#22-runtime-configuration-setup)
    - [2.2.1. Common Configurations](#221-common-configurations)
    - [2.2.2. Model-Specific Configurations](#222-model-specific-configurations)
  - [2.3. Cluster Configuration Setup](#23-cluster-configuration-setup)
  - [2.4. Quick Start](#24-quick-start)
  - [2.5. Benchmarking with `moe-permute-fusion`](#25-benchmarking-with-moe-permute-fusion)
  - [2.6. Benchmarking with `tp-comm-overlap`](#26-benchmarking-with-tp-comm-overlap)
- [3. DeepSeek Checkpoint Conversion](#deepseek-checkpoint-conversion)
  - [3.1. Checkpoint Conversion to Distributed Format](#41-checkpoint-conversion-to-distributed-format)
  - [3.2. DeepSeek-V2 Checkpoint Conversion](#42-deepseek-v2-checkpoint-conversion)
  - [3.3. DeepSeek-V3 Checkpoint Conversion](#43-deepseek-v3-checkpoint-conversion)

# 0. Container Setup
- Dockerfile: [dockers/Dockerfile](./dockers/Dockerfile).

# 1. Environment Setup

## 1.1. Login Node Setup
Before entering the container, to process the model configuration `.yaml` file using the `yq` package, follow these steps:

<details>
<summary>Click here to view steps.</summary>

1. Create a directory named `bin` in your home directory if it doesn't already exist:
    ```
    mkdir -p ~/.local/bin
    ```

2. Download the `yq` executable to the newly created directory:
    ```
    wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_amd64 -O ~/.local/bin/yq
    ```

3. Grant execution permissions to the `yq` executable:
    ```
    chmod +x ~/.local/bin/yq
    ```

4. Edit your `~/.bashrc` file and append the following line to include `~/.local/bin` in your system's `PATH`:
    ```
    export PATH="$HOME/.local/bin:$PATH"
    ```

5. Apply the changes by sourcing your shell configuration file:
    ```
    source ~/.bashrc
    ``` 

</details>

# 2. Performance Benchmarking

## 2.1. Test Script Setup
For performance benchmarking, you can launch scripts either using `sbatch` via [sbatch_benchmarking.sh](./sbatch_benchmarking.sh) or on an interactive node via [interactive_benchmarking.sh](./interactive_benchmarking.sh). 

- Environment Variable `MODEL`:

    The `MODEL` environment variable is required and must be explicitly defined in either the benchmarking script or the benchmarking command. We have predefined models including: `Mixtral-8x2B`, `Mixtral-8x7B`, `Mixtral-8x22B`, `DeepSeek-V2`, `DeepSeek-V2-Lite`, `DeepSeek-V3`, and `Qwen2-57B-A14B`.

- Environment Variables `CLUSTER`, `MCORE_RELEASE_VERSION`, and `MEGATRON_PATH`:

    These variables are required and must be explicitly defined in either the benchmarking script or the benchmarking command to ensure proper execution.

- Environment Variable `CONTAINER_IMAGE`:

    - The `CONTAINER_IMAGE` environment variable must be updated to either the path of your local container image or a Docker URL.
    - When using Gitlab Docker images, ensure that the port number is removed from the URL.
    - To import a container image into a local `.sqsh` file, use the following command:
        ```
		enroot import -o ./IMAGE.sqsh docker://[USER@][REGISTRY#]IMAGE[:TAG]
        ```
    - For more details, please refer to the [enroot documentation](https://github.com/NVIDIA/enroot/blob/master/doc/cmd/import.md).

- Using WandB for Experiment Tracking:

    - To utilize WandB for experiment tracking, replace `WANDB_API_KEY` with your own key from https://wandb.ai/authorize. It is highly recommended to add `export WANDB_API_KEY="your_own_wandb_api_key"` to your `~/.bashrc`.

## 2.2. Runtime Configuration Setup

### 2.2.1. Common Configurations
All common configurations can be adjusted either through [runtime_configs/benchmarking/common.conf](./runtime_configs/benchmarking/common.conf) or via the benchmarking command.

- Environment Variable `TRAINING_PARAMS_PATH`:

    To streamline the performance benchmarking process, we have provided preconfigured `.yaml` files for several commonly used MoE models, including [Mixtral-8x2B](./model_configs/benchmarking/Mixtral-8x2B.yaml), [Mixtral-8x7B](./model_configs/benchmarking/Mixtral-8x7B.yaml), [Mixtral-8x22B](./model_configs/benchmarking/Mixtral-8x22B.yaml), [DeepSeek-V2](./model_configs/benchmarking/DeepSeek-V2.yaml), [DeepSeek-V2-Lite](./model_configs/benchmarking/DeepSeek-V2-Lite.yaml), [DeepSeek-V3](./model_configs/benchmarking/DeepSeek-V3.yaml), [Qwen2-57B-A14B](./model_configs/benchmarking/Qwen2-57B-A14B.yaml). These files, located within the [megatron-moe-scripts/model_configs/benchmarking](./model_configs/benchmarking) directory, contain all the necessary configurations for the models.

- Environment Variable `COMMENT`:

    To append a comment to your `wandb-exp-name` for distinguishing it from other WandB experiments, please set `COMMENT` accordingly.

- Environment Variable `PROFILE`:

    To profile the training process, please set `PROFILE=1` when executing the benchmarking scripts.

- Environment Variable `PR`:
    
    For benchmarking with `fp8` data type, please set `PR=fp8` when launching the benchmarking scripts. Ensure that the container image is installed with TE version 1.7.0 or higher. For TE versions lower than 1.7.0, only the attention layer will be computed in `fp8`.

### 2.2.2. Model-Specific Configurations
All model-specific configurations can be adjusted either through [runtime_configs/benchmarking/runtime.conf](./runtime_configs/benchmarking/runtime.conf) or via the benchmarking command.

- Available Model-Specific Configurations:

    - Parallel Mappings: `TP`, `PP`, `EP`, `CP`, `VPP`, `PP_FIRST`, `PP_LAST`, and `LAYERS_PER_VP`.
    - Batch Sizes: `MBS` and `GBS`.
    - Model architecture: `NUM_LAYERS`.
    - MoE configurations: `MOE_TOKEN_DISPATCHER`, `MOE_GROUPED_GEMM`, and `--moe-extended-ep`.
    - Training configurations: `NNODES`, `RUN_TIME`, and `PRETRAIN`. Note that specifying a shorter running time may improve your job's priority in the Slurm queue.
    - Data configurations: `SEQ_LEN` and `DATASET`.

- Preconfigured Benchmarking Models:

    <details>
    <summary>Click here to view preconfigured benchmarking models.</summary>

    | Model | TP | PP | EP | CP | VPP | MBS | GBS | LAYERS | DISPATCHER | GROUPED_GEMM | NODES | RUN_TIME | PRETRAIN | SLEN | DATASET | PP_FIRST | PP_LAST |
    |----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
    | Mixtral-8x2B | 1 | 1 | 8 | 1 | 1 | 2 | 256 | 24 | alltoall | false | 8 | 00:20:00 | 1 | 4096 | Slimpajama | | |
    | Mixtral-8x7B | 1 | 4 | 8 | 1 | 8 | 1 | 256 | 32 | alltoall | true | 8 | 00:20:00 | 0 | 4096 | Slimpajama | | |
    | Mixtral-8x22B | 2 | 8 | 8 | 1 | 1 | 1 | 256 | 56 | alltoall | true | 16 | 00:20:00 | 0 | 4096 | Slimpajama | | |
    | DeepSeek-V2 | 1 | 16 | 8 | 1 | 2 | 1 | 1024 | 60 | alltoall | true | 32 | 00:20:00 | 0 | 4096 | Slimpajama | 2 | 2 |
    | DeepSeek-V2-Lite | 1 | 1 | 8 | 1 | 1 | 1 | 512 | 27 | alltoall | true | 1 | 00:20:00 | 0 | 4096 | Slimpajama | | |
    | DeepSeek-V3 | 1 | 16 | 64 | 1 | 1 | 1 | 8192 | 61 | flex | true | 128 | 00:20:00 | 0 | 4096 | Slimpajama | 4 | 1 |
    | Qwen2-57B-A14B | 2 | 4 | 4 | 1 | 7 | 1 | 256 | 28 | alltoall | true | 8 | 00:20:00 | 0 | 4096 | Slimpajama | | |

    </details>

## 2.3. Cluster Configuration Setup
All cluster configurations can be customized either through `cluster_configs/benchmarking/your_own_cluster.conf` or via the benchmarking command. For guidance on creating your own cluster configurations, please refer to the template provided in [cluster_configs/benchmarking/template.conf](./cluster_configs/benchmarking/template.conf).

- Required cluster-specific Slurm settings: `ACCOUNT`, `PARTITION`, `RUN_NAME`, and `CONTAINER_MOUNTS`.
- Required cluster-specific paths: `OUTPUT_PATH`, `DATA_PATH`, `TOKENIZER_MODEL`, and `LOAD_PATH`.

## 2.4. Quick Start
- To benchmark the model with training from scratch using preconfigured parameters, execute the following command:
    ```
    # Example for DeepSeek-V2-Lite
    MODEL=DeepSeek-V2-Lite bash ./sbatch_benchmarking.sh
    ```

- To train the model using costum parameters, refer to the following command:
    ```
    # Example for DeepSeek-V2-Lite
    MODEL=DeepSeek-V2-Lite MCORE_RELEASE_VERSION=0.11.0 PR=bf16 PROFILE=1 TP=1 PP=1 EP=8 VPP=1 MBS=1 GBS=512 SEQ_LEN=4096 MOE_TOKEN_DISPATCHER=alltoall MOE_GROUPED_GEMM=true bash ./sbatch_benchmarking.sh --moe-extended-ep
    ```

- To monitor your jobs, use `squeue -u $USER` for a one-time status check, or `watch -n 1 squeue -u $USER` for continuous monitoring. For detailed logging information, refer to the WandB dashboard.

## 2.5. Benchmarking with `moe-permute-fusion`
The `moe-permute-fusion` feature is currently compatible only with TE version 2.1.0 or higher. For TE versions lower than 2.1.0, please comment out the corresponding line in the preconfigured `.yaml` files:
```
--moe-permute-fusion: true
```

## 2.6. Benchmarking with `tp-comm-overlap`
For MLM-main, `tp-comm-overlap` can be enabled with specifically installed TE. Note that this feature currently only works for dense layer blocks (e.g., self-attention layers) and is not yet compatible with MoE layers.

To install TE with UserBuffer support, execute the following commands:
```
NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install git+https://github.com/NVIDIA/TransformerEngine.git
```

# 3. DeepSeek Checkpoint Conversion
## 3.1. Checkpoint Conversion to Distributed Format
- The DeepSeek checkpoint conversion scripts are designed to work with `TP=1`.

-  Conversion to Distributed Checkpoints:

    By default, the scripts generate legacy checkpoints. To convert to distributed checkpoints, please follow these steps:

    - First, convert to legacy checkpoints.
    - Modify the following line in your `.yaml` configuration file, located within the [megatron-moe-scripts/model_configs/benchmarking](./model_configs/benchmarking) directory: 
        ```
        --load: /path/to/legacy/checkpoint
        ```
    - Add the following lines to your  `.yaml` configuration file:
        ```
        --ckpt-convert-save: /path/to/save/distributed/checkpoint
        --ckpt-convert-format: torch_dist
        ```
    - Run the benchmarking script once to complete the conversion.

## 3.2. DeepSeek-V2 Checkpoint Conversion
- Download Checkpoint:

    Download the [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2) or [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) checkpoint from [HuggingFace](https://huggingface.co/deepseek-ai).

- Update Encironment Variables:

    Update the following environment variables in [convert_deepseek_v2.sh](./ckpt_convert_scripts/DeepSeek-V2/convert_deepseek_v2.sh): `MODEL`, `MEGATRON_PATH`, `SOURCE_CKPT_PATH`, and `TARGET_CKPT_PATH`.

- Run Conversion Script:

    Execute the conversion script using the following command:
    ```
    # Example for DeepSeek-V2
    MODEL=DeepSeek-V2 bash ./ckpt_convert_scripts/DeepSeek-V2/convert_deepseek_v2.sh

    # Example for DeepSeek-V2-Lite
    MODEL=DeepSeek-V2-Lite bash ./ckpt_convert_scripts/DeepSeek-V2/convert_deepseek_v2.sh
    ```

## 3.3. DeepSeek-V3 Checkpoint Conversion
- Download Checkpoint:

    Download the [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) checkpoint from [HuggingFace](https://huggingface.co/deepseek-ai).

- Update Encironment Variables:

    Update the following environment variables in [convert_deepseek_v3.sh](./ckpt_convert_scripts/DeepSeek-V3/convert_deepseek_v3.sh): `MODEL`, `MEGATRON_PATH`, `SOURCE_CKPT_PATH`, and `TARGET_CKPT_PATH`.

- Run Conversion Script:

    Execute the conversion script using the following command:
    ```
    MODEL=DeepSeek-V3 bash ./ckpt_convert_scripts/DeepSeek-V3/convert_deepseek_v3.sh
    ```