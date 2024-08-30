import os
from locust import HttpUser, task

from locust import User
from locust.exception import LocustError

import time
from typing import Any, Callable

import test_grpc.experimental.gevent as grpc_gevent
from grpc_interceptor import ClientInterceptor

# patch grpc so that it uses gevent instead of asyncio
grpc_gevent.init_gevent()

from locust import task

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc


class _LocustInterceptor(ClientInterceptor):
    def __init__(self, environment, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.env = environment

    def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        call_details: grpc.ClientCallDetails,
    ):
        response = None
        exception = None
        start_perf_counter = time.perf_counter()
        response_length = 0
        try:
            response = method(request_or_iterator, call_details)
            response_length = response.result().ByteSize()
        except grpc.RpcError as e:
            exception = e

        self.env.events.request.fire(
            request_type="grpc",
            name=call_details.method,
            response_time=(time.perf_counter() - start_perf_counter) * 1000,
            response_length=response_length,
            response=response,
            context=None,
            exception=exception,
        )
        return response


class GrpcUser(User):
    abstract = True
    stub_class = None

    def __init__(self, environment):
        super().__init__(environment)
        for attr_value, attr_name in ((self.host, "host"), (self.stub_class, "stub_class")):
            if attr_value is None:
                raise LocustError(f"You must specify the {attr_name}.")

        self._channel = grpc.insecure_channel(self.host)
        interceptor = _LocustInterceptor(environment=environment)
        self._channel = grpc.intercept_channel(self._channel, interceptor)

        self.stub = self.stub_class(self._channel)


class TritonHTTPUser(HttpUser):
    host = ""
    @task
    def hello_world(self):
        url = os.getenv("TRITON_SERVER_HTTP_URL")
        self.client.post(
            url,
            json={
                "inputs": [
                    {
                        "name": "INPUT_TEXT",
                        "data": ["о дивный новый мир"],
                        "shape": [1, 1],
                        "datatype": "BYTES"
                    },
                ]
            }
        )


class TritonGrpcUser(GrpcUser):
    host = "localhost:50051"
    channel = grpc.insecure_channel(host)
    stub_class = service_pb2_grpc.GRPCInferenceServiceStub(host)

    @task
    def sayHello(self):
        request = service_pb2.ModelInferRequest()
        try:
            response = self.stub.ModelInfer(request)
            # Process response
        except grpc.RpcError as e:
            print(f"gRPC error: {e}")

