#!/bin/bash

[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1

[  -z "$HF_MODEL_ID"  ] && echo "HF_MODEL_ID environment variable must exist" && exit 1
MODEL_PATH=/snapshots/$HF_MODEL_ID
[ ! -d $MODEL_PATH ] && echo "$MODEL_PATH not found" && exit 1

cp -r $MODEL_PATH /tmp/model
unset HF_MODEL_ID
unset MODEL_PATH

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

/usr/local/bin/dockerd-entrypoint.sh \
serve  \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"


