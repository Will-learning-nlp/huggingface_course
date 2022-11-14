from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 训练 准备 工作

# 第一步 整理格式--原本是trainer自动实现的功能

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)
# ['labels', 'input_ids', 'token_type_ids', 'attention_mask']

# 第二步--定义几个重要的对象
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True,
    batch_size=8, collate_fn=data_collator
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=8, collate_fn=data_collator
)

# 为了快速检验 数据处理，可以 如下验视1个batch

for batch in train_dataloader:
    print({k:v.shape for k,v in batch.items()})
    break
# {'labels': torch.Size([8]), 'input_ids': torch.Size([8, 74]),
# 'token_type_ids': torch.Size([8, 74]), 'attention_mask': torch.Size([8, 74])}
# 注意实际的形状可能与教程中不同，这是因为 shuffle 设置为 True
# 并且 在 batch 内 padding 至最大长度

# 以上，完成了数据的 preprocessing

# 下面转向 模型，和之前 的 实例化启动相同
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=2)
# 为了训练顺利，现在将 我们上面自己处理的 batch 传入 模型

for batch in train_dataloader:
    outputs = model(**batch)
    print(outputs.loss, outputs.logits.shape)
    break

# 写训练loop之前还有2件事
# 1 准备 optimizer
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

# 2 准备learning rate scheduler
# 默认的 learning rate scheduler 是 linear decay（最大值从 5e-5 到0）
# 为了 正确的定义它，我们要确定 我们的训练步数，
# 也就是 epochs的数量 乘上 训练batches的数量（即我们的 training dataloader的长度）
# Trainer默认使用3个epochs，所以我们也取3

from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

print(num_training_steps)

# 训练循环
# GPU
import torch

device = torch.device("cuda" if torch.cuda.is_available() else
                      torch.device("cpu"))
model.to(device)

print(device)

# 训练
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

model.train() # 此代码受下面受循环的控制吗
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        print(loss)
        print(loss.backward())

        print(optimizer.step())
        print(lr_scheduler.step())
        print(optimizer.zero_grad())
        progress_bar.update(1)
        break

# evaluation loop
# 使用  add_batch()方法  metrics累计batches，累计所有batches后，就可以
# 得到 最终的metric_compute()结果，方法如下

import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval() # 此代码受下面受循环的控制吗
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    print(logits)
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions,
                     references=batch["labels"])
    print(predictions)
    break
print(metric.compute())


# 上述 训练循环 在 单个 CPU和GPU上运行，
# 使用 库Accelerate 调参，可以在 多个 GPU OR TPU上训练
# 从 创建 training dataloader \ validation dataloader 开始调整
# 下面是如何实现--见10_fine_tuning_accelerate