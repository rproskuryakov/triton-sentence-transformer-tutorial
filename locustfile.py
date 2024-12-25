from locust import task, tag
from locust.env import Environment
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc

from src import grpc_user


class TritonGrpcUser(grpc_user.GrpcUser):
    host = "localhost:8001"
    stub_class = service_pb2_grpc.GRPCInferenceServiceStub

    def on_start(self):
        # Read custom argument from the command line
        self.model_version = self.environment.parsed_options.model_version

    @tag("ensemble_request")
    @task
    def ensemble_request(self):
        # model infer example
        # https://github.com/triton-inference-server/client/blob/519124f9e1ea938efffd23b435681b5e57df9ec0/src/python/examples/grpc_client.py#L94C5-L113C47
        self.stub.ModelInfer(
            service_pb2.ModelInferRequest(
                model_name="multilingual-e5-large-ensemble",
                model_version="v3",
                id="0",
                inputs=[
                    service_pb2.ModelInferRequest().InferInputTensor(
                        name="INPUT_TEXT",
                        datatype="TYPE_STRING",
                        shape=[-1],
                    ),
                ],
                outputs=[
                    service_pb2.ModelInferRequest().InferRequestedOutputTensor(
                        name="OUTPUT",
                        datatype="FP32",
                    )
                ],
                raw_input_contents=[bytes(1072812 * "a", "utf-8")]
            )
        )

    @tag("model_request")
    @task
    def model_request(self):
        # model infer example
        # https://github.com/triton-inference-server/client/blob/519124f9e1ea938efffd23b435681b5e57df9ec0/src/python/examples/grpc_client.py#L94C5-L113C47
        self.stub.ModelInfer(
            service_pb2.ModelInferRequest(
                model_name="multilingual-e5-large-onnx",
                model_version="v3",
                id="0",
                inputs=[
                    service_pb2.ModelInferRequest().InferInputTensor(
                        name="attention_mask",
                        datatype="TYPE_INT64",
                        shape=[-1],
                    ),
                    service_pb2.ModelInferRequest().InferInputTensor(
                        name="input_ids",
                        datatype="TYPE_INT64",
                        shape=[-1],
                    ),
                ],
                outputs=[
                    service_pb2.ModelInferRequest().InferRequestedOutputTensor(
                        name="sentence_embedding",
                        datatype="FP32",
                    )
                ],
                raw_input_contents=[bytes(1072812 * "a", "utf-8")]
            )
        )



# Add custom arguments to Locust CLI
def add_custom_arguments(parser):
    parser.add_argument("--model-version", type=str, default="v1", help="Triton model version.")


# Hook into locust's events to add custom arguments
def on_init(environment, **kwargs):
    add_custom_arguments(environment.parser)

Environment.on_init.add_listener(on_init)