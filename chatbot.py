'''import nltk
import tflearn
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import json
import tensorflow
import random

with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

print(data["intents"])

words = []
labels = []
docs_x = []
docs_y = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tags"])

    if intent["tags"] not in labels:
        labels.append(intent["tags"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net , 8)
net = tflearn.fully_connected(net , 8)
net = tflearn.fully_connected(net , len(output[0]), activation = "softmax")
model = tflearn.DNN(net)

model.fit(training,output, n_epoch= 1000, show_metric=True)
model.save("model.tflearn")'''
import random

import nltk

nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy
import json
import pickle
from tensorflow import keras
from sklearn.model_selection import train_test_split

stemmer = LancasterStemmer()
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

X_train, X_test, y_train, y_test = train_test_split(training, output, test_size=0.2, random_state=42)

model = keras.Sequential()
model.add(keras.layers.Input(shape=(len(training[0]),)))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(len(output[0]), activation='softmax'))

try:
    model.load('my_model.keras')
except:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save('my_model.keras')


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return bag  # Remove the wrapping list.


def chat():
    print("start talking with the bot!")
    while True:
        inp = input("You! ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]

        # Find the intent with the highest probability
        max_prob_index = numpy.argmax(results)
        max_prob = results[max_prob_index]

        if max_prob > 0.7:
            tag = labels[max_prob_index]

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("Sorry, I didn't understand your question. Please try again.")

        '''results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tag['responses']

            print(random.choice(responses))
        else:
            print("Sorry! I didn't get that, try again.")'''

chat()
