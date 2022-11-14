
# 用 pip install datasets 安装，
# ！！！要把 在 nlp 环境的 terminal 中运行
# 在 nlp 的 .env 中安装 不可用
# 开始 练习时 原以为 有必要 新建一个环境，现在看了没必要，还增加了麻烦
# ！！！所以，如果 在 一个环境中 又 创建 一个环境
# ！！！外部环境 安装的库 内部环境 都可以使用，但是 反之不行
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

# 可以看到 label标签 已经是 整数了，所以不必再 预处理
# 查看 整数 对应的 标签，可以直接 使用 features方法
raw_train_dataset = raw_datasets['train']
print(raw_train_dataset[0]) # 一行：包括1个句子对；还有标签信息

print(raw_train_dataset.features)
# 对应关系 names=['not_equivalent', 'equivalent'] --> [0, 1]

raw_val_ds = raw_datasets['validation']
print(raw_val_ds[15])
raw_ts_ds = raw_datasets['test']
print(raw_ts_ds[87])

from transformers import AutoTokenizer
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_s1 = tokenizer(raw_train_dataset['sentence1'][:2]) # 一列句子？
print('sentence1 有多少句子？')
print(len(raw_train_dataset['sentence1']))

tokenized_s2 = tokenizer(raw_train_dataset['sentence2'][:2])
print(tokenized_s1)
print('='*100)
print(tokenized_s2)
print('='*100)
print('='*100)
'''
注意 这里的 token_type_ids，它 告诉 模型 input 中 哪一部分是 第一列句子，哪一部分是 第二列句子 
2列分开 分别 处理，与 一起放到 tokenizer里面 同时处理的差别
同时处理会把所有对应的的第一句 第二句 合并 到一个列表向量里。
'''
inputs = tokenizer(raw_train_dataset['sentence1'][:2],raw_train_dataset['sentence2'][:2])
print(inputs)

decode_inp = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(decode_inp)

# map的使用
def tokenize_function(example):
    # example  是一个 字典，就是典型的数据集格式
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])

batch = data_collator(samples)
pad_dict = {k: v.shape for k, v in batch.items()}
print(pad_dict)
















