FROM nvcr.io/nvidia/tritonserver:25.03-py3

RUN pip install torch && \
    pip install --no-cache-dir accelerate==0.27.2 transformers==4.40.0

CMD ["tritonserver", "--model-repository=/models/"]