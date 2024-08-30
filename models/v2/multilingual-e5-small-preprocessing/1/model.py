from transformers import AutoTokenizer
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self._tokenizer = AutoTokenizer.from_pretrained(
            pb_utils.get_model_dir(),
            use_fast=True,
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            input_ = pb_utils.get_input_tensor_by_name(request, "text")
            input_string = input_.as_numpy()
            requests_texts = [i[0].decode("utf-8") for i in input_string]

            batch = self._tokenizer(
                requests_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="np",
            )
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "output_input_ids",
                            batch["input_ids"],
                        ),
                        pb_utils.Tensor(
                            "output_attention_mask",
                            batch["attention_mask"],
                        )
                    ]
                )
            )

        return responses
