from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
test_string = ["hello, world!",]
batch = tokenizer(test_string, max_length=512, padding=True, truncation=True, return_tensors='np')
print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
with open("models/v3/multilingual-e5-large-onnx/warmup/1/raw_input_ids", "wb") as fh:
    fh.write(batch["input_ids"])

with open("models/v3/multilingual-e5-large-onnx/warmup/1/raw_attention_mask", "wb") as fh:
    fh.write(batch["attention_mask"])


batch = tokenizer(test_string * 256, max_length=512, padding=True, truncation=True, return_tensors='np')
print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
with open("models/v3/multilingual-e5-large-onnx/warmup/256/raw_input_ids", "wb") as fh:
    fh.write(batch["input_ids"])

with open("models/v3/multilingual-e5-large-onnx/warmup/256/raw_attention_mask", "wb") as fh:
    fh.write(batch["attention_mask"])
