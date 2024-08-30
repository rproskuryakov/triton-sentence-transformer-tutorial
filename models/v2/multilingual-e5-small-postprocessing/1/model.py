# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
import torch
from torch.nn import functional
from torch.utils import dlpack


def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


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
                                    average_pool(
                                        dlpack.from_dlpack(
                                            pb_utils.get_input_tensor_by_name(
                                                request,
                                                "input_last_hidden_state",
                                            ).to_dlpack(),
                                        ),
                                        dlpack.from_dlpack(
                                            pb_utils.get_input_tensor_by_name(
                                                request,
                                                "input_attention_mask",
                                            ).to_dlpack(),
                                        ),
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

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        self.tokenizer = None
