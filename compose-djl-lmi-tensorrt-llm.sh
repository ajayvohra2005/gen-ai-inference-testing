#!/bin/bash

[ $# -ne 1 ] && echo "usage: $0 <up/down>" && exit 1

export IMAGE="deepjavalibrary/djl-serving:0.29.0-tensorrt-llm"
export COMMAND="/scripts/djl-lmi-tensorrt-llm.sh meta-llama/Meta-Llama-3-8B-Instruct"

if [ "$1" == "up" ]
then
sudo rm -rf  $HOME/djl-lmi
mkdir -p $HOME/djl-lmi

mkdir -p $HOME/scripts/djl-lmi
cp djl-lmi-tensorrt-llm.sh $HOME/scripts/djl-lmi/
chmod a+x $HOME/scripts/djl-lmi/*.sh

docker compose -f compose-djl-lmi-cuda.yaml up -d 
else
docker compose -f compose-djl-lmi-cuda.yaml down 
fi