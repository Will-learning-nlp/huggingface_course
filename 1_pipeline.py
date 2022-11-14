from transformers import pipeline

classifier = pipeline("sentiment-analysis")
# # classifier = pipeline("distilbert-base-uncased-finetuned-sst-2-english")
ootcome1 = classifier("it's cold out there")
print(ootcome1)

#
classifier = pipeline("zero-shot-classification")
outcome2 = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

print(outcome2)

generator = pipeline("text-generation")
generator_otc = generator("this algorithm is too complicated")
print(generator_otc)

#
generator = pipeline("text-generation", model="distilgpt2")
gener_otc = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(gener_otc)

unmasker = pipeline("fill-mask")
um_otc = unmasker("This course will teach you all about <mask> models.", top_k=3)
print(um_otc)

# unmasker_bert_base_cased = pipeline('bert-base-cased')
# un_otc_bb = unmasker_bert_base_cased("This course will teach you all about <mask> models.", top_k=3)

ner = pipeline("ner", grouped_entities=True)
ner_otc = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(ner_otc)

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])