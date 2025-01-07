#this file includes all the process to prepare the sata(tokenize, lemmatize)
import random
import json
import pickle
import numpy as np
import nltk     #Le module NLTK est une boîte à outils massive, destinée à vous aider avec l'ensemble de la méthodologie de traitement du langage naturel (NLP).
nltk.download('punkt_tab')
import tensorflow
from nltk.stem import WordNetLemmatizer #if we have working -work ... are ggoing to considered the same
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD   #stochastic rediant descent

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
intents = json.load(open("intent.json", "r", encoding='utf-8'))  # Added encoding for French text
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

# Process intents
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = word_tokenize(pattern.lower())  # Added lowercase for consistency
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

x_train = np.array([item[0] for item in training])
y_train = np.array([item[1] for item in training])

# Neural Network
model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation="softmax"))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5")
print("done")


