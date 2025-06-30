# triton-sentence-transformer-tutorial

# Deploying a Sentence Transformer with Triton Inference Server

This repository contains all the necessary code and scripts to deploy a huggingface retrieval model
such as `multilingual-e5-large` using NVIDIA's Triton Inference Server.
The guide covers every step from model export, configuration, and optimization
to deploying the model on Triton for high-performance inference.

## Contents

* Configuration files;
* Docker setup for Triton Server;
* Load testing code for sending inference requests.


A complete step-by-step guide is available in my detailed blog post:
[Deploying a Sentence Transformer with Triton Inference Server](https://rproskuryakov.github.io/posts/triton-sentence-transformer/).
This post explains the deployment process and how to use the files provided in this repository.

## How to Use

Clone this repository:

```commandline
git clone git@github.com:rproskuryakov/triton-sentence-transformer-tutorial.git
cd triton-sentence-transformer-tutorial
```

Then build a docker image with the following command
```commandline
docker build . -t tritonserver-tutorial
```

To launch an instance of triton inference server you need to add model files to each of the iterations, v1, v2 and v3.
For the first case download the model files with huggingface-cli
```commandline
huggingface-cli download --local-dir $(pwd)/models/v1/multilingual-e5-large/1/ \
  intfloat/multilingual-e5-large
```

For second one install the optimum package and convert the model to onnx with optimum-cli
```commandline
pip install optimum[exporters,onnxruntime]
optimum-cli export onnx --model intfloat/multilingual-e5-large \
    --task feature-extraction  --library-name sentence_transformers \
     --framework pt  models/v2/multilingual-e5-large-onnx/1/
```
Also move tokenizer files to preprocessing model folder
```commandline
mv ./models/v2/multilingual-e5-large-onnx/1/*token* ./models/v2/multilingual-e5-large-preprocessing/1/
mv ./models/v2/multilingual-e5-large-onnx/1/sentencepiece.bpe.model ./models/v2/multilingual-e5-large-preprocessing/1/
```

For the third configuration do the same as for the second one.

Once all file is placed, to launch a docker container with one of the configurations, execute
```commandline
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
    -v "$(pwd)/models/v1/:/models/" tritonserver-tutorial \
    tritonserver --model-repository=/models
```

Change -v "$(pwd)/models/v1/:/models/" according to the version you'd like to use.



## License
This project is licensed under the MIT License.

downloading a model from hfhub

```commandline
huggingface-cli download --local-dir $(pwd)/models/v1/multilingual-e5-large/1/ \
  intfloat/multilingual-e5-large
```
```commandline
docker build . -t tritonserver:1.0.0
make build
make deploy
make load-test


```

docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
    -v "$(pwd)/models/v1/:/models/" tritonserver:1.0.0 \
    tritonserver --model-repository=/models
## Delete

```commandline
docker pull nvcr.io/nvidia/tritonserver:25.03-py3-sdk
```

scp -r ./* noisyrave@62.84.126.125:~/project