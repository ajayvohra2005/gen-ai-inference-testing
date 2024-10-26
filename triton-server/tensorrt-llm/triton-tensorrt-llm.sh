#!/bin/bash


export BUILD_SCRIPT="/scripts/build-engine.sh"
export MODEL_NAME="llama3_8b_instruct_tensorrt_llm"
export MODEL_SERVER_CORES=8
/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"