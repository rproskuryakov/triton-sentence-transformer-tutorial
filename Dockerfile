FROM nvcr.io/nvidia/tritonserver:24.05-py3

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --pre torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir accelerate==0.27.2 transformers==4.40.0

RUN tritonserver \
		--model-repository=/models/$(VERSION) \
		--log-verbose=1