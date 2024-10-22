# triton-sentence-transformer-tutorial

# Deploying a Model with Triton Inference Server
This repository contains all the necessary code and scripts to deploy a huggingface retrieval model
such as `multilingual-e5-large` using NVIDIA's Triton Inference Server.
The guide covers every step from model export, configuration, and optimization
to deploying the model on Triton for high-performance inference.

# Contents

* Model conversion scripts (e.g., ONNX or TensorFlow to Triton format)
* Configuration files (config.pbtxt)
* Docker setup for Triton Server
* Load testing code for sending inference requests


A complete step-by-step guide is available in my detailed blog post:
[Deploying a Sentence Transformer with Triton Inference Server](https://rproskuryakov.github.io/posts/triton-sentence-transformer/).
This post explains the deployment process and how to use the files provided in this repository.

# How to Use

Clone this repository:

```bash
git clone git@github.com:rproskuryakov/triton-sentence-transformer-tutorial.git
cd triton-sentence-transformer-tutorial
```

Follow the instructions in the guide to set up Triton and deploy your model.

# License
This project is licensed under the MIT License.

[//]: # (# Terraform start)

[//]: # ()
[//]: # (https://github.com/Paperspace/terraform-provider-paperspace?ref=blog.paperspace.com)

[//]: # ()
[//]: # (https://youtu.be/P3__yTs24rU)

[//]: # ()
[//]: # (poetry init)

[//]: # ()
[//]: # (```)

[//]: # (docker run -p 8089:8089 -v $PWD:/mnt/locust locustio/locust -f /mnt/locust/locustfile.py)

[//]: # (```)
