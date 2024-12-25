FROM nvcr.io/nvidia/tritonserver:24.11-py3

RUN pip install torch && \
    pip install --no-cache-dir accelerate==0.27.2 transformers==4.40.0

ENTRYPOINT tritonserver \
		--model-repository=/models/$(VERSION) \
		--log-verbose=1