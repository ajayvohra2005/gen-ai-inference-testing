#!/bin/bash

[ ! -d /djl-lmi ] && echo "/djl-lmi dir must exist" && exit 1
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1

[ $# -ne 1 ] && echo "usage: $0 hf-model-id" && exit 1 

MODEL_PATH=/snapshots/$1
[ ! -d $MODEL_PATH ] && echo "$MODEL_PATH not found" && exit 1

LOG_ROOT=/djl-lmi/logs
CACHE_DIR=/djl-lmi/cache
mkdir -p $CACHE_DIR

cat > /opt/ml/model/serving.properties <<EOF
option.model_id=$MODEL_PATH
option.entryPoint=djl_python.transformers_neuronx
option.tensor_parallel_degree=8
option.amp=f16
option.n_positions=8192
option.model_loading_timeout=1800
option.model_loader=tnx
option.rolling_batch=auto
option.rolling_batch_strategy=continuous_batching
option.max_rolling_batch_size=8
option.output_formatter=json
option.trust_remote_code=true

EOF

mkdir -p $LOG_ROOT
OUTPUT_LOG="$LOG_ROOT/djl-lmi-server.log"
export NEURON_CC_FLAGS="--model-type transformer"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
/usr/local/bin/dockerd-entrypoint.sh \
serve \
2>&1 | tee $OUTPUT_LOG \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"

