# Gen AI Inference Examples

This tutorial shows Gen AI inference server examples using [Deep Java Library (DJL) Large Model Inference (LMI) Server](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html), and [Triton Inference Server](https://github.com/triton-inference-server). 

## Tutorial Steps

### Launch Deep Learning Ubuntu Desktop

This tutorial assumes a `trn1.32xlarge` machine for Neuron examples, and a `g5.48xlarge` for CUDA examples. You may want to launch the [AWS Deep Learning Desktop](https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop) with  `trn1.32xlarge` instance type for neuron, or `g5.48xlarge` for gpu.

### Build containers

To build the container for triton inference server with neuronx, execute this on `trn1.32xlarge` machine:

    ./scripts/build-tritonserver-neuronx.sh

To build the container for triton inference server with cuda and vLLM, execute this on gpu machine:

    ./scripts/build-tritonserver-cuda-vllm.sh

To build the container for triton inference server with cuda and TensorRT-LLM, execute this on gpu machine::

    ./scripts/build-tritonserver-cuda-trtllm.sh
    
### Download Hugging Face Model

To download a Hugging Face model (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`):

    ./scripts/hf-snapshot.sh hf-model-id hf-token

The model is downloaded under `$HOME/snapshots`

### Deep Java Library (DJL) Serving Large Model Inference (LMI) Transformers Neuronx Engine

This test should be run on a `trn1.32xlarge instance`. To launch DJL Serving with LMI Transformers Neuronx engine:

    ./djl-serving/transformers-neuronx/compose-djl-lmi-transformers-neuronx.sh up

To test:

    ./djl-serving/tests/test-djl-lmi.sh
    ./djl-serving/tests/test-djl-lmi-concurrent.sh

To stop DJL LMI server:

    ./djl-serving/transformers-neuronx/compose-djl-lmi-transformers-neuronx.sh down

### Triton Inference Server with vLLM and Neuronx

To launch Triton Inference Server with vLLM and Neuronx, execute this on `trn1.32xlarge`:

    ./triton-server/vllm/compose-triton-vllm-neuronx.sh up

To test:

    ./triton-server/tests/test-triton-vllm.sh
    ./triton-server/tests/test-triton-vllm-concurrent.sh

To stop the server:

    ./triton-server/vllm/compose-triton-vllm-neuronx.sh down

### Triton Inference Server with DJL-LMI Transformers Neuronx Engine

To launch Triton Inference Server with DJL-LMI Transformers Neuronx engine:

    ./triton-server/djl-lmi/compose-triton-djl-lmi-neuronx.sh up

To test:

    ./triton-server/tests/test-triton-djl-lmi-neuronx.sh
    ./triton-server/tests/test-triton-djl-lmi-neuronx-concurrent.sh

To stop the server:

    ./triton-server/djl-lmi/compose-triton-djl-lmi-neuronx.sh down
