# 1. 数据集
from datasets import load_dataset
raw_datasets = load_dataset("conll2003")
print(raw_datasets)

# 这里的文本是词汇list，最后1列称为 tokens
# 但是这些词 是 pre-tokenized inputs，还需要经过tokenizer做进一步的
# subword tokenization

# 查看训练集的第一个元素
print('查看训练集的第一个元素、第二个元素')
print(raw_datasets["train"][0]["tokens"])
print(raw_datasets["train"][1]["tokens"])
# 查看 NER 标签
print("查看ner标签")
print(raw_datasets["train"][0]["ner_tags"])
print(raw_datasets["train"][1]["ner_tags"])
# ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
# ['Peter', 'Blackburn']
# [3, 0, 7, 0, 0, 0, 7, 0, 0]
# [1, 2]

# 这些就是 用来训练的整数标签，但是 当我们 验视 数据时不一定有用
# 对于文本分类，我们可以 获取 这些整数 和 标签名字 的相似性（对应关系），
# 方式是 查看 我们数据集的 特征属性

ner_feature = raw_datasets["train"].features["ner_tags"]
print(ner_feature)
# Sequence(feature=ClassLabel(names=['O', 'B-PER', 'I-PER', 'B-ORG',
# 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
# id=None), length=-1, id=None)

# 所以这一列 包括 ClassLabels 序列的 元素
# 可以通过 feature 的name属性查看ner名字
label_names = ner_feature.feature.names
print(label_names)
# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']


# 下面decoding

words = raw_datasets["train"][4]["tokens"]
labels = raw_datasets["train"][4]["ner_tags"]
print("第5句词的labels")
print(labels)
line1= ""
line2= ""
# 巧妙的 对齐代码 ----效果 就是 词与标签 格式 长度上 严格对齐
for word, label in zip(words, labels):
    full_label = label_names[label]  #
    max_length = max(len(word), len(full_label))  # 取 词的长度 和 词标签长度 中 较长的1个
    line1 += word + " " * (max_length - len(word) + 1)  # 算 最大长度 去除 词所占长度 再加 1个空格
    line2 += full_label + " " * (max_length - len(full_label) + 1)  # 5-5+1=1，只有1个空格的长度
print(line1)
print(line2)

"""
#  对 pos 标签 尝试 同样的 词与标签 对齐 操作
words = raw_datasets["train"][4]["tokens"]
labels = raw_datasets["train"][4]["pos_tags"]
pos_feature = raw_datasets["train"].features["pos_tags"]
label_names = pos_feature.feature.names
line1= ""
line2= ""
# 巧妙的 对齐代码 ----效果 就是 词与标签 格式 长度上 严格对齐
for word, label in zip(words, labels):
    full_label = label_names[label]  # labels 是 元素为数字的list，这里是 用数字 索引 名称元素
    max_length = max(len(word), len(full_label))  # 取 词的长度 和 词标签长度 中 较长的1个
    line1 += word + " " * (max_length - len(word) + 1)  # 算 最大长度 去除 词所占长度 再加 1个空格
    line2 += full_label + " " * (max_length - len(full_label) + 1)  # 5-5+1=1，只有1个空格的长度
print(line1)
print(line2)
"""

# 2. 处理数据
# 文本 要 转为 model 可以处理的 IDs
# 分类 任务 要有 pre-tokenized inputs
# tokenize API 可以实现这一需求；
# 注意 参数-- a special flag-- is_split_into_words=True
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# 这里 我们 可以用 Hub 中的任何 模型 取代 model_checkpoint
# 或者 本地存储的 预训练模型和tokenizer
# 唯一的限制是 tokenizer 需要 被 huggingface 库 Tokenizers 支持
# 所以 有 快速 版本，是否支持 只需查看 tokenizer的is_fast属性
print(tokenizer.is_fast)

# 处理1句测试
print("处理1句测试，查看tokenizer处理后的inputs")
inputs = tokenizer(raw_datasets["train"][0]["tokens"],
                   is_split_into_words=True)
print(inputs)  # 一个字典， 值是3个list
print(inputs.tokens())  # 具体的 切分词， 字符串列表list；
# 注意到 lamb 分为 la， mb，这时词数增加，如何对齐标签？
# 库 Tokenizers 可以实现
print(inputs.word_ids())

# 下面具体 实现 分词（subword）后的 复杂 对齐

def align_labels_with_tokens(labels, word_ids):
    '''
    :param labels: 词 对应的 标签的 数字索引，为list
    如：
    第5句词的labels
    [5, 0, 0, 0, 0, 3, 4, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0]
    :param word_ids: list，元素为 自然数，是词的序号，包括开头结尾的none
    如   [None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]
    :return:
    '''
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # 只要 新的 id 与 cur当前id不同，必然换词了
            # 一个新词的开始
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # 特殊符号 -- special token
            # The first rule we’ll apply is that special tokens get a label of -100.
            # This is because by default -100 is an index that is ignored in the loss
            # function we will use (cross entropy).
            new_labels.append(-100)
        else:
            # 此时 这个 sub词 与前一个 sub词 同属1个单词
            label = labels[word_id]
            # 并且 修改 B-XXX 为 I-XXX
            if label % 2 == 1:  # 因为B-XXX索引都是奇数
                label += 1
                # 因为 对应的 I-XXX 就在 B-XXX的后面，
                # 比如 ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
            new_labels.append(label)
    return new_labels


labels = raw_datasets["train"][0]["ner_tags"]  # 这是原生数据，数字元素的list
word_ids = inputs.word_ids()  # 这是用 tokenizer处理后的数据，数字元素的list
print(labels)
print(align_labels_with_tokens(labels,word_ids))
# [3, 0, 7, 0, 0, 0, 7, 0, 0]
# [-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]
# 注意到 lamb 本来1个0，对齐后 2个0
# 并且 增加了 两个 -100，是 句子开头结尾 的特殊符号

"""
Some researchers prefer to attribute only one label per word, 
and assign -100 to the other subtokens in a given word. 
This is to avoid long words that split into lots of subtokens 
contributing heavily to the loss. Change the previous function
 to align labels with input IDs by following this rule.
"""
# ：另一种方式--
# 每个词 1个标签，其余的 同属1个词的 sub词 都标记为 -100
# 避免 太长的词 切出 一堆，影响loss
# 只要 改一行代码 应该就可以：else: label = -100

# 要 预处理 我们的整个 数据集，需要tokenize 所有的输入，并应用上面的对齐函数
# 要同时处理的话，使用 Dataset.map()方法，并且设置 batched=True
# 唯一的不同是 word_ids()函数要 获得 列表的列表
# 具体实现如下


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True,
        is_split_into_words=True
    )
    all_labels = examples["ner_tags"]  # ner标签列的所有行
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)  # 取i对应的 句子的 词 的 ids
        new_labels.append(align_labels_with_tokens(labels, word_ids))
        # 处理 当前第i句 的对齐；合并到总list--new_labels中
    tokenized_inputs["labels"] = new_labels
    # 将上述 循环所有 行 得到的 对齐列表，存入 字典 tokenizerd_inputs；
    return tokenized_inputs
    # tokenizerd_inputs 是一个 字典 dict
    # 比如 第一句话处理后 为 {'input_ids': [101, 7270, 22961, 1528, 1840, 1106, 21423, 1418, 2495, 12913, 119, 102],
    # 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    # 所以此处 tokenized_inputs["labels"] = new_labels 是 新增加 键值对

# 注意到，我们还没有 padding 输入，随后在 用 data collator 创建 batchs 会padding

# 现在，总的处理一下我们的数据
# Dataset.map()方法；此方法对 数据集 的 每个元素 施加 一个函数，


tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    # .column_names 获得了 所有的 列名称；
    # 所以，原始数据集 raw_datasets中 所有的列 都删除；
    # 新的 tokenized_datasets 中全是tokenize后的数据
)
print(tokenized_datasets)

"""
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 14041
    })
    validation: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 3250
    })
    test: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 3453
    })
})
"""
# 所以以上的效果：数据集行数不变，函数中第一次处理的input_ids被新的 分词后的input_ids取代，
# 相应的features中的labels 被新的 对应 新的 分词ids 的 labels取代

# 以上为 preprocessing，是 最困难的 部分，下面的训练就和第三章差不多

# 3. fine-tuning
# 使用trainer的代码和之前一样，
# 唯一的变化是 数据collate 成为 batch的方式 变化
# 和 metric compute_function 的变化

# 3.1 data collation
# 这里不能 像第3章中 那样直接使用 DataCollatorWithPadding
# 因为 那样 只能 pad inputs（inputIDs, attention mask, token IDs）
# 这里 的 labels 要 与 inputs 以 相同的方式 pad，以保证它们 尺寸 一致
# 使用 -100 实现 loss 计算 忽略的效果

# 以上功能 由 DataCollatorForTokenClassification实现，

from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# 使用 几个 样本 做测试

batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
print(batch["labels"])
# tensor([[-100,    3,    0,    7,    0,    0,    0,    7,    0,    0,    0, -100],
#         [-100,    1,    2, -100, -100, -100, -100, -100, -100, -100, -100, -100]])

# 下面输出  数据集 中 labels，对比一下
for i in range(2):
    print(tokenized_datasets["train"][i]["labels"])
# [-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]
# [-100, 1, 2, -100]
# 可以发现，第二个list被padded为与第一个长度相同的list，借助 -100

# 3.2 metrics
# 要使 Trainer 计算 每个 epoch 中的 metric，
# 我们需要定义 1个 compute_metrics（）函数，
# 此函数 接收 predictions 和 labels 数组，
# 返回 字典，其中有 metric 和values
# 用于 评估 token classification prediction 的传统 框架 是
# seqeval，已用pip安装至本环境

# 然后 可以 通过 evaluate.load()函数 加载 此 框架
import evaluate

metric = evaluate.load("seqeval")
# 这个 metric 不像 标准准确度 一样， 它 接收 字符串形式的 标签列表，而不是整数；
# 所以 我们 需要 完全地 decode predictions 和 labels，再把它们 传入 metric
#
# 第一步，我们 先获得 第一组训练例子的 labels
labels = raw_datasets["train"][0]["ner_tags"]
labels = [label_names[i] for i in labels]
print(labels)  # 注意，中间测试pos没改引用名（现已注释），导致出错
# ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
#
# 第二步，通过改变index2的值，来 创建 虚拟的 predictions
predictions = labels.copy()
predictions[2] = "O"  # 注意，是字母，不是数字0；
a_test_metric = metric.compute(predictions=[predictions], references=[labels])
# references 写错字母，检查5分钟！！！
print(a_test_metric)

# {'MISC': {'precision': 1.0, 'recall': 0.5, 'f1': 0.6666666666666666, 'number': 2},
# 'ORG': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1},
# 'overall_precision': 1.0,
# 'overall_recall': 0.6666666666666666,
# 'overall_f1': 0.8,
# 'overall_accuracy': 0.8888888888888888}

# 可以看到，返回了很多信息，有每个实体的 准确率，召回率，F1分数，以及总体的
# 我们只保留 总体分数
# 此 compute_metrics()函数 首先 将 logits的argmax 转为 predictions（
# logits 和 probabilities 有相同的 次序，所以不需要应用 softmax）
# 然后 我们 将 labels 和 predictions 由 整数 转为 字符串。
# 我们 将 所有标签为-100的值删除，
# 然后将结果 传给 metric_compute()函数

import numpy as np


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # logits  什么形状，
    print(logits)
    predictions = np.argmax(logits, axis=-1)
    # np.argmax()是numpy中获取array的某一个维度中数值最大的那个元素的索引
    # numpy.argmax() 和 numpy.argmin()函数分别沿给定轴返回最大和最小元素的索引。

    # 删除 忽略的 index（special tokens），转为labels
    true_labels = [[label_names[l] for l in label if l != -100]
                   for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]  # 取出 预测 整数 对应的 真实的字符串 结果
    all_metrics = metric.compute(predictions=true_predictions,
                                 references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

# 现在 Trainer 马上完成，还需要1个模型

# 3.3 定义 模型
# 注意 传入 标签数量
# 设定 正确的 标签对应关系
# 借助2个字典，如下，它们包含了 从ID到label的映射mappings


id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

# 现在 将它们 传入 模型，它们会被设置为模型config，然后保存并上传到Hub

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
# 与第3章中类似，这里会报warning--一些weights 没使用，一些weights随机初始化
# 即 原预训练head删除，新创建 本次训练head
# 但是，先检查一下模型的标签数量
print(model.config.num_labels)  # 9

# 注意，这里的 模型标签数量 检查 非常重要，如果模型标签数量不对
# 会在 Trainer.train()时报错，

# 3.4 fine-tuning模型
# 定义 Train儿 前 最后2件事：登录hugging face & 定义参数

# from huggingface_hub import notebook_login
# notebook_login()

# 使用方法2，在终端输入命令 huggingface-cli login
# 注意 token 登录时，输入或者粘贴不显示，但是已经输入进去了
# 结果如下
"""
Login successful
Your token has been saved to /Users/glad/.huggingface/token
Authenticated through git-credential store but this isn't the helper defined on your machine.
You might have to re-authenticate when pushing to the Hugging Face Hub.
Run the following command in your terminal in case you want to set this credential helper as the default

git config --global credential.helper store
(nlp) glad@Beihus-MacBook-Air ~ % 
"""

# define TrainingArguments

from transformers import TrainingArguments
args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

# 3.4 启动训练

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()

# 此处开始训练后，报错 没有 git lfs，pip install git lfs后依然报错
# 检查 安装文件在，版本也是最新
# 先尝试在base 和 本nlp环境中 都安装，依然报错
# 尝试初始化 git lfs install，报错
# 更新 git 网页端 文件夹，依然报错
# 官方 下载 安装包，不知道.sh的使用方法，以为无效，
# brew 失败，404
# csdn查询到 cloning 方法，翻墙网络下 下载， cloning文件30万量集，依然报错
# 最后，直接在terminal运行.sh文件，成功初始化git，训练正常进行。


#  训练开始，每个epoch模型会保存一次，并上传到Hub
#  所以 我们能够在另一个机器上恢复训练

trainer.push_to_hub(commit_message="Training complete")








