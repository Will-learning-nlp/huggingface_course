# 总结7中代码
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 训练测试
# 传参数--创建 超参数对象
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
# If you want to automatically upload your model to the Hub during training,
# pass along push_to_hub=True in the TrainingArguments.

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer
trainer = Trainer(model, training_args, train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["validation"],
                  data_collator=data_collator,
                  tokenizer=tokenizer,)
                    #
                    # #data_collator used by the Trainer will be a DataCollatorWithPadding as defined previously,
                    # so you can skip the line data_collator=data_collator in this call. I

# 训练
# trainer.train()

# 评估 准确率
# 使用上面训练结果，将 验证数据集 传入模型 训练， 获得 预测数据
predictions = trainer.predict(tokenized_datasets['validation'])
print(predictions.predictions.shape, predictions.label_ids.shape)

# 预测数据转换，以方便比较
import numpy as np
preds = np.argmax(predictions.predictions, axis=-1)

import evaluate

metric = evaluate.load("glue", "mrpc")
metr_compu = metric.compute(predictions=preds, references=predictions.label_ids)

print(metr_compu)

# 集成上述内容，得到下面函数，用于评估模型预测表现
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 新的 训练器

training_args =TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer =Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,

)

trainer.train()













