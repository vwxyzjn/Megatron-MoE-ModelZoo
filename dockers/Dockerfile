FROM nvcr.io/nvidia/pytorch:25.02-py3 as base

# Build the image
# nvidia-docker build --target base -f dockers/Dockerfile --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" --tag <IMAGE_TAG> --network host .

ENV SHELL /bin/bash

RUN rm /opt/megatron-lm -rf
RUN apt-get update
RUN apt-get install -y sudo gdb pstack bash-builtins git zsh autojump tmux curl
RUN pip install debugpy dm-tree torch_tb_profiler einops wandb
RUN pip install sentencepiece tokenizers transformers torchvision ftfy modelcards datasets tqdm pydantic
RUN pip install nvidia-pytriton py-spy yapf darker pytest-cov pytest_mock

# envsubst used for model_params substitution
RUN apt-get install -y gettext

# yq is a lightweight and portable command-line YAML processor; install yq through snapd
RUN wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

RUN pip3 install --no-cache-dir \
    black==24.4.2 \
    isort==5.13.2 \
    flake8==7.1.0 \
    pylint==3.2.6 \
    coverage \
    mypy

# Install grouped_gemm
RUN TORCH_CUDA_ARCH_LIST="8.0 9.0" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
# Install TE w/ NVTE_WITH_USERBUFFERS=1 so that `--tp-comm-overlap` can be enabled
RUN NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.2

RUN pip install setuptools==69.5.1

# Install DeepEP
## Add ssh private key into container so we can access gitlab
ARG SSH_PRIVATE_KEY
RUN mkdir ~/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" > ~/.ssh/id_rsa
RUN chmod 600 ~/.ssh/id_rsa
RUN ssh-keyscan bitbucket.org >> ~/.ssh/known_hosts
RUN ssh-keyscan -H <GITLAB_HOST> >> ~/.ssh/known_hosts

## Install gdrcopy
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/gdrcopy.git
WORKDIR /tmp/gdrcopy
RUN git checkout v2.4.1

RUN apt update
RUN apt install -y nvidia-dkms-535
RUN apt install -y build-essential devscripts debhelper fakeroot pkg-config dkms
RUN apt install -y check libsubunit0 libsubunit-dev

WORKDIR /tmp/gdrcopy/packages
RUN CUDA=/usr/local/cuda ./build-deb-packages.sh
RUN dpkg -i gdrdrv-dkms_*.deb
RUN dpkg -i libgdrapi_*.deb
RUN dpkg -i gdrcopy-tests_*.deb
RUN dpkg -i gdrcopy_*.deb

ENV GDRCOPY_HOME=/usr/src/gdrdrv-2.4.1/

## the dependency of IBGDA
RUN ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so
RUN apt-get install -y libfabric-dev

## Clone and build deepep and deepep-nvshmem
WORKDIR /home/dpsk_a2a
RUN git clone https://github.com/deepseek-ai/DeepEP.git
### Note: This NVSHMEM repo needs to be created following the instructions in DeepEP's third-party documentation
### https://github.com/deepseek-ai/DeepEP/blob/main/third-party/README.md
RUN git clone <DEEPEP_NVSHMEM_REPO_URL>
RUN rm ~/.ssh/id_rsa

## Build deepep-nvshmem
WORKDIR /home/dpsk_a2a/deepep-nvshmem
ENV CUDA_HOME=/usr/local/cuda
RUN NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/home/dpsk_a2a/deepep-nvshmem/install -DCMAKE_CUDA_ARCHITECTURES=90 && cd build && make install -j

## Build deepep
WORKDIR /home/dpsk_a2a/DeepEP
ENV NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem/install
RUN NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem/install python setup.py develop
RUN NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem/install python setup.py install