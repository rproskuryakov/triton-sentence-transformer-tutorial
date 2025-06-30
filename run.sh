#!/bin/bash
echo $GPU
echo $MODEL_SET_VERSION
if [[ $GPU == "TRUE" ]]
then
  sudo docker run -d --gpus=1 --rm --runtime=nvidia -p8000:8000 -p8001:8001 -p8002:8002 \
  -v ./models/$MODEL_SET_VERSION/:/models/ \
  --name triton-inference-server-$MODEL_SET_VERSION \
  triton-custom:1.0.0
elif [[ $GPU == "FALSE" ]]
then
  sudo docker run -d --rm --runtime=nvidia -p8000:8000 -p8001:8001 -p8002:8002 \
  -v ./models/$MODEL_SET_VERSION/:/models/ \
  --name triton-inference-server-$MODEL_SET_VERSION \
  triton-custom:1.0.0 \
  tritonserver --model-repository=/models/ \
	--model-config-name=cpu
else
	echo "GPU variable must be set to TRUE or FALSE"
fi

dpkg -i cuda-repo-ubuntu2404-X-Y-local_12.9*_x86_64.deb


