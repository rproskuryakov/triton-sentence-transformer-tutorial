import torch
from transformers import pipeline
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        self._model = pipeline(
            "feature-extraction",
            model="intfloat/multilingual-e5-large",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            input_ = pb_utils.get_input_tensor_by_name(request, "input_text")
            input_string = input_.as_numpy()
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "output_embeddings",
                            F.normalize(
                                torch.Tensor(
                                    self._model(
                                        [i[0].decode("utf-8") for i in input_string],
                                    ),
                                ).mean(dim=1),
                                p=2,
                                dim=1,
                            ),
                        ),
                    ]
                )
            )

        return responses
