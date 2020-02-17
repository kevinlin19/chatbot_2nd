import json
import os
import pickle
import random
from os.path import exists
from plistlib import load
from turtle import Shape

import keras
import nltk
import numpy as np
import tensorflow as tf

stemmer = nltk.LancasterStemmer()

with open('intents.json') as file:
        data = json.load(file)

try:
    with open('data.pickle', 'rb') as file:
        words, tags, train_x, train_y = pickle.load(file)
except:
    words = []
    tags = []
    doc_x = [] # for training model
    doc_y = [] # for training model
    for intent in data['intents']:
        for pattern in intent['patterns']:
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            doc_x.append(word)
            doc_y.append(intent['tag'])

        if intent['tag'] not in tags:
            tags.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in ["?", "!", "."]] # stemming word
    words = sorted(list(set(words)))

    tags = sorted(tags)

    # onehot encoding sample
    output_empty = [0] * len(tags)

    # training set
    train_x = []
    train_y = []

    # training input into onehot encoding
    for xi, doc in enumerate(doc_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        # onehot
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output = output_empty[:]
        output[tags.index(doc_y[xi])] = 1

        # assign to train_x and train_y
        train_x.append(bag)
        train_y.append(output)
    # list to array
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    with open('data.pickle', 'wb') as file:
        pickle.dump((words, tags, train_x, train_y), file)

if not os.path.exists('model.bin'):
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, input_shape=(len(train_x[0]),)))
    model.add(keras.layers.Dense(8))
    model.add(keras.layers.Dense(len(train_y[0]), activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=1000, batch_size=16)
    # save
    model.save('model.bin')

else:
    model = keras.models.load_model('model.bin')
    # print(model.summary())



def bag_of_input(inp, word_lib):
    bag = [0] * len(word_lib)

    inp_wrds = nltk.word_tokenize(inp)
    inp_wrds = [stemmer.stem(w.lower()) for w in inp_wrds]

    for inp_w in inp_wrds:
        for idx, w in enumerate(word_lib):
            if inp_w == w:
                bag[idx] = 1

    return np.array(bag)

def chatbot():
    print("Let talk to AI bot! (press 'quit' to exit!)")
    name = input("What's your name? ")

    while True:
        inp = input(f"{name}: ")
        if inp.lower() == 'quit':
            break
        result = model.predict_classes(bag_of_input(inp, words).reshape(1, -1))

        tag = tags[result[0]]

        for tag_search in data['intents']:
            if tag_search['tag'] == tag:
                response = tag_search['response']
                break
        print(random.choice(response))


chatbot()
