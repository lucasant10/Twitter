import sys
sys.path.append('../')
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
import numpy as np
from sklearn.metrics import accuracy_score
import configparser
import h5py

class PoliticalClassification:

    def __init__(self, arg_model, dictfile, maxlen):
        cf = configparser.ConfigParser()
        cf.read("../file_path.properties")
        path = dict(cf.items("file_path"))
        self.dir_model = path['dir_model']
        self.maxlen = maxlen
        self.model = load_model(self.dir_model + arg_model)
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.vocab = np.load(self.dir_model + dictfile).item()
        

    def is_political(self, tweet):
        X = list()
        seq = list()
        for word in tweet.split(' '):
            seq.append(self.vocab.get(word, self.vocab['UNK']))
        X.append(seq)
        data = pad_sequences(X, maxlen= self.maxlen)
        y_pred = self.model.predict(data)
        y_pred = np.argmax(y_pred, axis=1)
        return True if y_pred == 0 else False
        