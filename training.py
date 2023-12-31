import random
import json
import numpy as np
import pickle

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation ,Dropout
from tensorflow.keras.optimizers import SGD



intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

for intent in intents['intents']:
  for pattern in intent['patterns']:
    word_list = nltk.word_tokenize(pattern)
    words.append(word_list)
    documents.append((word_list, intent['tag']))
    if intent['tag'] not in classes:
      classes.append(intent['tag'])

lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word) for word_list in words for word in word_list if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pickle', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
  bag = []
  words_patterns = documents[0][0]  # Get the list of words from the first document
  words_patterns = [lemmatizer.lemmatize(word.lower()) for word in words_patterns]

  for word in words:
    bag.append(1) if word in words_patterns else bag.append(0)



  output_row = list(output_empty)
  output_row[classes.index(document[1])] = 1
  training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),),activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64 , activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',optimizer = sgd , metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5, verbose=1)

model.save('chatbot_model.model')
print("Done")