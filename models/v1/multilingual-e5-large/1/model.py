import torch

import triton_python_backend_utils as pb_utils
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        self._tokenizer = AutoTokenizer.from_pretrained(pb_utils.get_model_dir())
        self._model = AutoModel.from_pretrained(pb_utils.get_model_dir())
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_ = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            input_texts = [i[0].decode("utf-8") for i in input_.as_numpy()]
            batch_dict = self._tokenizer(
                input_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = self._model(**batch_dict)
            tensor = pb_utils.Tensor(
                            "OUTPUTS",
                            F.normalize(
                            average_pool(outputs.last_hidden_state, batch_dict["attention_mask"]),
                            p=2, dim=1).cpu().detach().numpy()                          ,
                        )
            print(tensor, flush=True)
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        tensor,
                    ]
                )
            )

        return responses
