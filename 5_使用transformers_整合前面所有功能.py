from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs)


sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
model_inputs_1 = tokenizer(sequences)
print(model_inputs_1)

# Will pad the sequences up to the maximum sequence length
# 批量 所有句子 填充到 最长句子的长度
model_inputs_2 = tokenizer(sequences, padding="longest")
print(model_inputs_2)

# 批量 所有句子 填充的 模型可处理的 最大长度
# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs_3 = tokenizer(sequences, padding="max_length")
print(model_inputs_3)

# 自定义padding长度
# Will pad the sequences up to the specified max length
model_inputs_4 = tokenizer(sequences, padding="max_length", max_length=8)
print(model_inputs_4)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Will truncate the sequences that are longer than the model max length
# 删节比模型最大处理长度 长 的序列
# (512 for BERT or DistilBERT)
model_inputs_5 = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
# 删节比 自定义 最大处理长度 长 的序列
model_inputs_6= tokenizer(sequences, max_length=8, truncation=True)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Returns PyTorch tensors
model_inputs_7 = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
# model_inputs_8 = tokenizer(sequences, padding=True, return_tensors="tf")
# ImportError: Unable to convert output to TensorFlow tensors format, TensorFlow is not installed.

# Returns NumPy arrays
model_inputs_9 = tokenizer(sequences, padding=True, return_tensors="np")

#注意 下面 直接 tokenizer 一键输出 和手动分步 输出 的结果差异

sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
# [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]

# 开头与结尾各多出一个，这是什么呢
#
# 解码一下，看看字符串是什么
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))

# "[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
# "i've been waiting for a huggingface course my whole life."

# The tokenizer added the special word [CLS] at the beginning and the special word [SEP] at the end. This is because the model was pretrained with those, so to get the same results for inference we need to add them as well。
# 当然，不同的模型处理不同
# import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

###### 总结
# 如何处理 多序列、长序列、多类型tensors输出选择
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)