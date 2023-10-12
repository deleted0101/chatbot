import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import json
#import tflearn
import tensorflow
import random

with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

print(data["intents"])

words = []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)

    if intent["tags"] not in labels:
        labels.append(intent["tags"])



