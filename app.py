import random
import json
import pickle
import numpy as np
import nltk     #Le module NLTK est une boîte à outils massive, destinée à vous aider avec l'ensemble de la méthodologie de traitement du langage naturel (NLP).
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model



lemmatizer = WordNetLemmatizer()  #instance de l'objet WordNetLemmatizer de la bibliothèque NLTK (Natural Language Toolkit). Cette classe est utilisée pour le lemmatiser des mots, c'est-à-dire, réduire les mots à leur forme de base (ou racine).
intents=json.load(open("intent.json", "r"))

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))


model = load_model("chatbot_model.h5")

#creer de fcts to clean up the sentences, predicting the class based on the sentence   #fct pour la reponse
# chatbot.py
def clean_up_phrase(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())  # Added lowercase
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words          # #retourne un  tableau des elements apres filtration(tokenized and lemmatized) donnee par l utilisateur


def bag_of_words(sentence):
    sentence_words = clean_up_phrase(sentence)
    bag = [0] * len(words)
    for s_word in sentence_words:
        for i, word in enumerate(words):
            if s_word == word:   #si on trouve un mot de l user similaire a celui dans la bd on met le tab a 1
                bag[i] = 1
    return np.array(bag)

#fct de prediction
def predict_class(sentence):
    bow = bag_of_words(sentence)     #car elle doit etre donnee au reseau neuronal sous format numerique
    res = model.predict(np.array([bow]))[0]              #res is the output of the model prediction, which is an arr of probabilities for each class
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]  #i : indice of each class, the proba of each  class

    if not results:  # If no results above threshold
        return [{"intent": "unknown", "probability": "0"}]

    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def get_response(intents_list, intents_json):
    if intents_list[0]['intent'] == "unknown":
        return "Je ne comprends pas votre question. Pouvez-vous reformuler ?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json["intents"]

    for i in list_of_intents:
        if i["tag"] == tag:
            return random.choice(i["responses"])

    return "Je ne comprends pas votre question. Pouvez-vous reformuler ?"

print("GOO!! CEentIA is running..")

while True:
    message=input("")
    ints=predict_class(message)  #par les le cnn
    res=get_response(ints,intents)
    print(res)
