from transformers import AutoTokenizer

checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
# return_tensors 参数指定我们要返回的tensor类型--PyTorch, TensorFlow, or plain NumPy

from transformers import AutoModel

model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

print(outputs.logits.shape)
print(outputs.logits)
import torch
# 将 logits 转为 概率
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
# 将概率转为可读文字标签结果--报错？？？
# outcome = model.config.id2label
# model.config.id2label
# print(outcome)