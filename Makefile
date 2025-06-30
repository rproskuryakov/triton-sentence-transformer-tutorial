MODEL_SET_VERSION=v2
N_CONCURRENT_REQUESTS=50
TRITON_HOSTNAME=localhost:8000
RELEASE=25.03
GPU=FALSE
MODEL_NAME=intfloat/multilingual-e5-large
export GPU
export MODEL_SET_VERSION


build:
	sudo docker buildx build -t triton-custom:1.0.0 .

deploy:
	eval './run.sh $(GPU) $(MODEL_SET_VERSION)'

# launch load-testing for particular model and num of concurrent requests
load-test:
	sudo docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
	sudo docker run --rm -it --net host -v./testing/:/testing/ \
		nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk \
		perf_analyzer -m multilingual-e5-large \
		--concurrency-range ${N_CONCURRENT_REQUESTS}:${N_CONCURRENT_REQUESTS} \
		--measurement-interval=30000 \
		--input-data=/testing/input_data.json \
		-b 1 \
		-f /testing/reports/latency_results_${MODEL_SET_VERSION}.csv \
		--verbose-csv

pull-model:
	hgf download

export2onnx:
	optimum-cli export onnx --model multilingual-e5-large/ \
                        --task feature-extraction \
                        --library-name sentence_transformers \
                        --framework pt \
                        onnx_model/
	mv -T $(pwd)/models/v2/multilingual-e5-large-onnx/1 \
		$(pwd)/onnx_model/config.json \
		$(pwd)/onnx_model/model.onnx \
		$(pwd)/onnx_model/model.onnx_data
	mv -T $(pwd)/models/v2/multilingual-e5-large-preprocessing/1 \
		$(pwd)/onnx_model/tokenizer_config.json \
		$(pwd)/onnx_model/tokenizer.json \
		$(pwd)/onnx_model/special_tokens_map.json \
		$(pwd)/onnx_model/sentencepiece.bpe.model
	rm -rf onnx_model/
