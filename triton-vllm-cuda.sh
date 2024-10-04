#!/bin/bash

[ ! -d /triton ] && echo "/triton dir must exist" && exit 1
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1

[ $# -ne 1 ] && echo "usage: $0 hf-model-id" && exit 1 

HF_MODEL_ID=$1
MODEL_PATH=/snapshots/$HF_MODEL_ID
[ ! -d $MODEL_PATH ] && echo "$MODEL_PATH not found" && exit 1

LOG_ROOT=/triton/logs
MODEL_REPO=/triton/model_repository

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
  "max_num_seqs": 8,
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
tritonserver \
--model-repository=${MODEL_REPO} \
--grpc-port=8001 \
--http-port=8000 \
--metrics-port=8002 \
--disable-auto-complete-config \
--log-file=$OUTPUT_LOG \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"