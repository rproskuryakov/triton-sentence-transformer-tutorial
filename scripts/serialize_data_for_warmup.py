from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
test_strings = ["hello, world!", "bye, world!"]
batch = tokenizer(test_strings, max_length=512, padding=True, truncation=True, return_tensors='np')
print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
with open("models/v3/multilingual-e5-large-onnx/warmup/raw_input_ids", "wb") as fh:
    fh.write(batch["input_ids"])

with open("models/v3/multilingual-e5-large-onnx/warmup/raw_attention_mask", "wb") as fh:
    fh.write(batch["attention_mask"])
