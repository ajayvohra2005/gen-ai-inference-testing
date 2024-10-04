#!/bin/bash


[ $# -ne 1 ] && echo "usage: $0 <up/down>" && exit 1
export IMAGE=
export COMMAND="/scripts/triton-vllm-neuronx.sh meta-llama/Meta-Llama-3-8B-Instruct"

if [ "$1" == "up" ]
then
sudo rm -rf  $HOME/triton0
sudo rm -rf  $HOME/triton1
sudo rm -rf  $HOME/triton2
sudo rm -rf  $HOME/triton3

mkdir -p $HOME/triton0
mkdir -p $HOME/triton1
mkdir -p $HOME/triton2
mkdir -p $HOME/triton3

mkdir -p $HOME/scripts/triton
cp triton-vllm-neuronx.sh $HOME/scripts/triton/
cp vllm-neuron-issue-1.patch $HOME/scripts/triton/
chmod a+x $HOME/scripts/triton/*.sh

mkdir -p $HOME/cache

docker compose -f compose-triton-neuronx.yaml up -d 
else
docker compose -f compose-triton-neuronx.yaml down
fi
