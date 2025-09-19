# syntax=docker/dockerfile:experimental
# Use Python 3.10 as the base image
FROM nvcr.io/nvidia/pytorch:25.06-py3

# Install system dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y curl gnupg

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk git

WORKDIR /home/megatron-moe
COPY . /home/megatron-moe
# Set the default Python version to 3.10
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1