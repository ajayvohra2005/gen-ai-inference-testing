#!/bin/bash

[ ! -d /djl-lmi ] && echo "/djl-lmi dir must exist" && exit 1
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1

[ $# -ne 1 ] && echo "usage: $0 hf-model-id" && exit 1 

MODEL_PATH=/snapshots/$1
[ ! -d $MODEL_PATH ] && echo "$MODEL_PATH not found" && exit 1

cp -r $MODEL_PATH /tmp/model

LOG_ROOT=/djl-lmi/logs
CACHE_DIR=/djl-lmi/cache
mkdir -p $CACHE_DIR

cat > /opt/ml/model/serving.properties <<EOF
option.model_id=/tmp/model
option.entryPoint=djl_python.tensorrt_llm
option.tensor_parallel_degree=4
option.max_num_tokens=8192
option.dtype=fp16
option.rolling_batch=trtllm
option.max_rolling_batch_size=4
option.output_formatter=json
option.trust_remote_code=true

EOF

mkdir -p $LOG_ROOT
OUTPUT_LOG="$LOG_ROOT/djl-lmi-server.log"
/usr/local/bin/dockerd-entrypoint.sh \
serve \
2>&1 | tee $OUTPUT_LOG \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"

