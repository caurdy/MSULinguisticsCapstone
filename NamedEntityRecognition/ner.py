import sys
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
# example = "My name is Wolfgang and I live in Berlin. My phone number is 2482770767"
#
# ner_results = nlp(example)
# print(ner_results)

def ner(text):
    sentence = text
    ner_result = nlp(sentence)
    return ner_result
# 12 sec for test 1 that has 98 sentences
# 17 sec for test 2 that has 134
# 18 sec for test 3 thgat has 181

if __name__ == '__main__':
    ner(sys.argv[1])