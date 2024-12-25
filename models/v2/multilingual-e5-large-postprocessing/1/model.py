# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
import torch
from torch.nn import functional
from torch.utils import dlpack


class TritonPythonModel:
    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(
                                "output_vector",
                                functional.normalize(
                                        torch.from_numpy(
                                            pb_utils.get_input_tensor_by_name(
                                                request,
                                                "input_last_hidden_state",
                                            ).as_numpy()
                                        ),
                                    p=2,
                                    dim=1,
                                ).cpu().detach().numpy(),
                            ),
                        ],
                    )
                )
            except Exception as e:
                print(e, flush=True)
        return responses
