from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
# model = BertModel(config)
print(config)

from transformers import BertConfig, BertModel

config = BertConfig()
# model = BertModel(config)

# Model is randomly initialized!

from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
# model.save_pretrained("directory_on_my_computer") # /Users/glad/python-study/翻译技术_NLP/transformers_saved_models
# model.save_pretrained("/Users/glad/python-study/翻译技术_NLP/transformers_saved_models") #

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sample_t = tokenizer("Using a Transformer network is simple")
print(sample_t)

# tokenization 分解步骤，原理展示
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)

in_ids = tokenizer.convert_tokens_to_ids(tokens)
print(in_ids)

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)

"""
{'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
[7993, 170, 13809, 23763, 2443, 1110, 3014]
Using a transformer network is simple
"""