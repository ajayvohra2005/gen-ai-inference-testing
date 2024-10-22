#!/bin/bash

[ ! -d /triton ] && echo "/triton dir must exist" && exit 1
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1

[ $# -ne 1 ] && echo "usage: $0 hf-model-id" && exit 1 

HF_MODEL_ID=$1
MODEL_PATH=/snapshots/$HF_MODEL_ID
[ ! -d $MODEL_PATH ] && echo "$MODEL_PATH not found" && exit 1

LOG_ROOT=/triton/logs/$HF_MODEL_ID
mkdir -p $LOG_ROOT

TP_SIZE=8
PP_SIZE=1

CKPT_PATH=$MODEL_PATH/trtllm_ckpt

if [ ! -d $CKPT_PATH ]
then
echo "Convert HF ckpt to TensorRT-LLM ckpt" 
GIT_CLONE_DIR=/tmp/trtllm
git clone https://github.com/NVIDIA/TensorRT-LLM.git $GIT_CLONE_DIR
cd $GIT_CLONE_DIR
git checkout main
git fetch origin 250d9c293d5edbc2a45c20775b3150b1eb68b364
git reset --hard 250d9c293d5edbc2a45c20775b3150b1eb68b364

OUTPUT_LOG=$LOG_ROOT/hf_to_trtllm.log
pip install --upgrade pip
pip install datasets==2.20.0
pip install evaluate~=0.4.2
pip install rouge_score~=0.1.2
pip install sentencepiece~=0.2.0

python3 \
examples/llama/convert_checkpoint.py \
--model_dir=$MODEL_PATH \
--output_dir=$CKPT_PATH \
--dtype=float16 \
--tp_size=$TP_SIZE \
2>&1 | tee $OUTPUT_LOG 
fi

GIT_CLONE_DIR=/tmp/trtllm_backend
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git $GIT_CLONE_DIR
cd $GIT_CLONE_DIR
git checkout main
git fetch origin 76464e9be06600f3979acad9c14857938a66ff9f
git reset --hard 76464e9be06600f3979acad9c14857938a66ff9f


mkdir -p /cache/$HF_MODEL_ID
ENGINE_DIR=/cache/$HF_MODEL_ID/trtllm_engine

if [ ! -d $ENGINE_DIR ]
then
echo "Build TensorRT-LLM engine"
OUTPUT_LOG=$LOG_ROOT/build_engine.log

trtllm-build \
--checkpoint_dir ${CKPT_PATH} \
--max_num_tokens 32768 \
--tp_size ${TP_SIZE} \
--pp_size ${PP_SIZE} \
--gpus_per_node 8 \
--remove_input_padding enable \
--gemm_plugin float16 \
--gpt_attention_plugin float16 \
--paged_kv_cache enable \
--context_fmha enable \
--output_dir ${ENGINE_DIR} \
--max_batch_size 8 \
--use_custom_all_reduce disable \
2>&1 | tee $OUTPUT_LOG 

mpirun --allow-run-as-root -np $TP_SIZE python3 examples/run.py --tokenizer_dir $MODEL_PATH --engine_dir $ENGINE_DIR --max_output_len 128
fi

MODEL_REPO=/tmp/trtllm_model_repo
mkdir -p $MODEL_REPO
MODEL_NAME=llama3_8b_instruct

echo "Build Triton TensorRT-LLM model"

OUTPUT_LOG=$LOG_ROOT/build_model.log
TOKENIZER_DIR=$MODEL_PATH
TOKENIZER_TYPE=auto
DECOUPLED_MODE=false
MAX_BATCH_SIZE=8
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=100
FILL_TEMPLATE_SCRIPT=tools/fill_template.py
cp -r all_models/inflight_batcher_llm/preprocessing $MODEL_REPO/${MODEL_NAME}_preprocessing
cp -r all_models/inflight_batcher_llm/postprocessing $MODEL_REPO/${MODEL_NAME}_postprocessing
cp -r all_models/inflight_batcher_llm/tensorrt_llm_bls $MODEL_REPO/${MODEL_NAME}_tensorrt_llm_bls
cp -r all_models/inflight_batcher_llm/ensemble $MODEL_REPO/${MODEL_NAME}_ensemble
cp -r all_models/inflight_batcher_llm/tensorrt_llm $MODEL_REPO/${MODEL_NAME}_tensorrt_llm
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}_preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT} 2>&1 | tee $OUTPUT_LOG
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}_postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT} 2>&1 | tee -a $OUTPUT_LOG
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},tensorrt_llm_model_name:${MODEL_NAME}_tensorrt_llm 2>&1 | tee -a $OUTPUT_LOG
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}_ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE} 2>&1 | tee -a $OUTPUT_LOG
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching 2>&1 | tee -a $OUTPUT_LOG
sed -i 's/name: "preprocessing"/name: "llama3_8b_instruct_preprocessing"/1' ${MODEL_REPO}/${MODEL_NAME}_preprocessing/config.pbtxt
sed -i 's/name: "postprocessing"/name: "llama3_8b_instruct_postprocessing"/1' ${MODEL_REPO}/${MODEL_NAME}_postprocessing/config.pbtxt
sed -i 's/name: "tensorrt_llm_bls"/name: "llama3_8b_instruct_tensorrt_llm_bls"/1' ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm_bls/config.pbtxt
sed -i 's/name: "ensemble"/name: "llama3_8b_instruct_ensemble"/1' ${MODEL_REPO}/${MODEL_NAME}_ensemble/config.pbtxt
sed -i 's/name: "tensorrt_llm"/name: "llama3_8b_instruct_tensorrt_llm"/1' ${MODEL_REPO}/${MODEL_NAME}_tensorrt_llm/config.pbtxt
sed -i 's/model_name: "preprocessing"/model_name: "llama3_8b_instruct_preprocessing"/1' ${MODEL_REPO}/${MODEL_NAME}_ensemble/config.pbtxt
sed -i 's/model_name: "postprocessing"/model_name: "llama3_8b_instruct_postprocessing"/1' ${MODEL_REPO}/${MODEL_NAME}_ensemble/config.pbtxt
sed -i 's/model_name: "tensorrt_llm"/model_name: "llama3_8b_instruct_tensorrt_llm"/1' ${MODEL_REPO}/${MODEL_NAME}_ensemble/config.pbtxt

echo "Start Triton Inference Server"

OUTPUT_LOG="$LOG_ROOT/triton_server.log"
python3 \
scripts/launch_triton_server.py \
--tensorrt_llm_model_name=llama3_8b_instruct_tensorrt_llm \
--world_size=$((TP_SIZE * PP_SIZE)) \
--model_repo=${MODEL_REPO} \
--grpc_port=8001 \
--http_port=8000 \
--metrics_port=8002 \
--log-file=$OUTPUT_LOG \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"