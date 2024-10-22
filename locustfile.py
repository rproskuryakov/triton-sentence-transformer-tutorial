from locust import task

from src import grpc_user
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc


class TritonGrpcUser(grpc_user.GrpcUser):
    host = "localhost:8001"
    stub_class = service_pb2_grpc.GRPCInferenceServiceStub

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
