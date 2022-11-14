# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
#
# tokenizer = GPT2Tokenizer.from_pretrained('/Users/glad/python-study/翻译技术_NLP/distilbert-base-uncased-finetuned-sst-2-english')
#
# text = 'this course is not as friendly as it claims'
#
# indexed_tokens = tokenizer.encode(text)
#
# token_tensor = torch.tensor([indexed_tokens])
#
# model = GPT2LMHeadModel.from_pretrained('/Users/glad/python-study/翻译技术_NLP/distilbert-base-uncased-finetuned-sst-2-english')
#
# model.eval()
#
# with torch.no_grad():
#     outputs = model(token_tensor)
#     predictions = outputs[0]
#
# predicted_index = torch.argmax(predictions[0:-1,:]).item()
#
# predicted_text = tokenizer.decode(indexed_tokens+[predicted_index])
#
# print(predicted_text)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")


# t_t = tokenizer('what is this course doing')
# print(t_t)