#!/bin/bash

[ ! -d /triton ] && echo "/triton dir must exist" && exit 1
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1

[ $# -ne 1 ] && echo "usage: $0 hf-model-id" && exit 1 

HF_MODEL_ID=$1
MODEL_PATH=/snapshots/$HF_MODEL_ID
[ ! -d $MODEL_PATH ] && echo "$MODEL_PATH not found" && exit 1

LOG_ROOT=/triton/logs
MODEL_REPO=/triton/model_repository
CACHE_DIR=/cache

GIT_CLONE_DIR=/tmp/vllm
git clone https://github.com/vllm-project/vllm.git $GIT_CLONE_DIR
cd $GIT_CLONE_DIR
git checkout main
git fetch origin 5b734fb7edfdf3f8a836a3ddee81eba506230fdd
git reset --hard 5b734fb7edfdf3f8a836a3ddee81eba506230fdd
git apply --ignore-whitespace /scripts/vllm-neuron-issue-1.patch

cat > /tmp/config.pbtxt <<EOF
backend: "vllm"
max_batch_size: 0
model_transaction_policy {
  decoupled: true
}

input [ 
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [1]
  },
  {
      name: "stream"
      data_type: TYPE_BOOL
      dims: [1]
      optional: true
  },
  {
      name: "sampling_parameters"
      data_type: TYPE_STRING
      dims: [1]
      optional: true
  },
  {
      name: "exclude_input_in_output"
      data_type: TYPE_BOOL
      dims: [1]
      optional: true
  }
] 
output [
  {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [-1]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_MODEL
  }
]

EOF

cat > /tmp/model.json <<EOF
{
  "model": "$MODEL_PATH",
  "disable_log_requests": true,
  "tensor_parallel_size": 8,
  "max_num_seqs": 4,
  "dtype": "float16",
  "max_model_len": 8192,
  "block_size": 8192,
  "use_v2_block_manager": true
}

EOF

mkdir -p $LOG_ROOT
OUTPUT_LOG="$LOG_ROOT/triton_server.log"
rm -rf $MODEL_REPO
mkdir -p $MODEL_REPO
VERSION=1
MODEL_NAME=llama3-8b-instruct
mkdir -p $MODEL_REPO/$MODEL_NAME/$VERSION
cp /tmp/model.json $MODEL_REPO/$MODEL_NAME/$VERSION/model.json
cp /tmp/config.pbtxt $MODEL_REPO/$MODEL_NAME/config.pbtxt
cd $GIT_CLONE_DIR
pip3 install -r requirements-neuron.txt
pip3 install .
pip3 install triton==2.2.0
pip3 install pynvml==11.5.3
git clone https://github.com/triton-inference-server/vllm_backend.git /tmp/vllm_backend
cd /tmp/vllm_backend
git fetch origin 507e4dccabf85c3b7821843261bcea7ea5828802
git reset --hard 507e4dccabf85c3b7821843261bcea7ea5828802

mkdir -p /opt/tritonserver/backends/vllm
cp -r /tmp/vllm_backend/src/* /opt/tritonserver/backends/vllm/
cd $GIT_CLONE_DIR
export NEURON_CC_FLAGS="--model-type transformer"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
export OMP_NUM_THREADS=32
tritonserver \
--model-repository=${MODEL_REPO} \
--grpc-port=8001 \
--http-port=8000 \
--metrics-port=8002 \
--disable-auto-complete-config \
--log-file=$OUTPUT_LOG \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"