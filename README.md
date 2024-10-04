# AWS Neuron Inference Server Examples

This tutorial shows [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/containers/index.html) inference server examples using [Deep Java Library (DJL) Large Model Inference (LMI) Server](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html), and [Triton Inference Server](https://github.com/triton-inference-server). 

The DJL-LMI server supports *continuous batching* for higher throughput. The Triton Inference Server supports *dynamic batching*. The custom Python backed `execute` function supports model batch size greater than 1 for higher throughput (default is set to 4).

## Tutorial Steps

### Launch AWS Neuron Ubuntu instance

This tutorial assumes a `trn1.32xlarge` machine for Neuron examples, and a `g5.48xlarge` for CUDA examples. You may want to launch the [AWS Deep Learning Desktop](https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop) with required instance type for a fully configured AWS Neuron EC2 desktop.

### Download Hugging Face Model

To download a Hugging Face model (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`):

    ./hf-snapshot.sh hf-model-id hf-token

The model is downloaded under `$HOME/snapshots`

### Deep Java Library (DJL) Large Model Inference (LMI) Server

To launch DJL LMI server:

    ./compose-djl-lmi-transformers-neuronx.sh up

This starts a single DJL-LMI server listening on host port 8000 that round-robins requests to 4 model instances running with the server.

To test:

    ./test-djl-lmi.sh
    ./test-djl-lmi-concurrent.sh

To stop DJL LMI server:

    ./compose-djl-lmi-transformers-neuronx.sh down

### Triton Inference Server with vLLM and Neuronx

First we need to build the required docker image, and push it into ECR in your AWS region. To build and push the image:

    git clone \
    https://github.com/aws-samples/amazon-eks-machine-learning-with-terraform-and-kubeflow.git
    cd amazon-eks-machine-learning-with-terraform-and-kubeflow
    ./containers/tritonserver-neuronx/build_tools/build_and_push.sh aws-region

Next, set the URI of the image built above in the `IMAGE` variable in [compose-triton-vllm-neuronx.sh](./compose-triton-vllm-neuronx.sh). 

To launch Triton Inference Server with a custom Python backend:

    ./compose-triton-vllm-neuronx.sh up

On `trn1.32xlarge`, this starts 4 instances of Triton Inference Server, listening on ports 8000, 8010, 8020 and 8030. You may want to use an AWS Application Load Balancer in front of it if you want a single endpoint.

To test:

    ./test-triton-vllm.sh
    ./test-triton-vllm-concurrent.sh

To stop the server:

    ./compose-triton-vllm-neuronx.sh down

### Triton Inference Server with Deep Java Library Python Backend with Transformers Neuronx

First we need to build the required docker image, and push it into ECR in your AWS region. To build and push the image:

    git clone \
    https://github.com/aws-samples/amazon-eks-machine-learning-with-terraform-and-kubeflow.git
    cd amazon-eks-machine-learning-with-terraform-and-kubeflow
    ./containers/tritonserver-neuronx/build_tools/build_and_push.sh aws-region

Next, set the URI of the image built above in the `IMAGE` variable in [compose-triton-djl-python-neuronx.sh](./compose-triton-djl-python-neuronx.sh). 

To launch Triton Inference Server with a custom Python backend:

    ./compose-triton-djl-python-neuronx.sh up

On `trn1.32xlarge`, this starts 4 instances of Triton Inference Server, listening on ports 8000, 8010, 8020 and 8030. You may want to use an AWS Application Load Balancer in front of it if you want a single endpoint.

To test:

    ./test-triton-djl-python.sh
    ./test-triton-djl-python-concurrent.sh

To stop the server:

    ./compose-triton-djl-python-neuronx.sh down
