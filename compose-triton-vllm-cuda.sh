#!/bin/bash

[ $# -ne 1 ] && echo "usage: $0 <up/down>" && exit 1

export IMAGE="nvcr.io/nvidia/tritonserver:24.06-vllm-python-py3"
export COMMAND="/scripts/triton-vllm-cuda.sh meta-llama/Meta-Llama-3-8B-Instruct"

if [ "$1" == "up" ]
then
sudo rm -rf  $HOME/triton0


mkdir -p $HOME/triton0

mkdir -p $HOME/scripts/triton
cp triton-vllm-cuda.sh $HOME/scripts/triton/
chmod a+x $HOME/scripts/triton/*.sh

mkdir -p $HOME/cache

docker compose -f compose-triton-cuda.yaml up -d 
else
docker compose -f compose-triton-cuda.yaml down
fi
