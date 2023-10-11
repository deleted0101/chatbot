import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import json
import tflearn
import tensorflow
import random


with open("intents.json") as file:
    data = json.load(file)

print(data["intents"])