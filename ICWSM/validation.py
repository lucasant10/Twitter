import sys
sys.path.append('../')
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
import numpy as np
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
from keras.utils import np_utils
import operator
import gensim, sklearn
from collections import defaultdict
from batch_gen import batch_gen
import os
import configparser
from text_processor import TextProcessor
import json
import h5py


def select_tweets(tweets, tw_class):
    # selects the tweets as in mean_glove_embedding method
    # Processing       
    X, Y = [], []
    tweet_return = []
    tweet_class = []
    for i, tweet in enumerate(tweets):
        _emb = 0
        for w in tweet:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb += 1
        if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
            tweet_class.append(tw_class[i])
    print('Tweets selected:', len(tweet_return))
    return tweet_return, tweet_class

def load_files(dir_in):
    doc_list = list()
    tw_files = sorted([file for root, dirs, files in os.walk(dir_in)
                 for file in files if file.endswith('.json')])
    tw_class = list()
    for tw_file in tw_files:
        temp = list()
        with open(dir_in+tw_file) as data_file:
            for line in data_file:
                tweet = json.loads(line)
                temp.append(tweet['text'])
                doc_list.append(tweet['text'])
                tw_class.append(tw_file.split(".")[0])
    return doc_list, tw_class

def gen_sequence():
    y_map = dict()
    for i, v in enumerate(sorted(set(tw_class))):
        y_map[v] = i
    print(y_map)
    X, y = [], []
    for i, tweet in enumerate(tweets):
        seq, _emb = [], []
        for word in tweet:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(y_map[tw_class[i]])
    return X, y



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN based models for politics twitter')
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-m', '--modelfile', required=True)
    parser.add_argument('-d', '--dictfile', required=True)
    parser.add_argument('-l', '--maxlen', required=True)

    
    args = parser.parse_args()

    W2VEC_MODEL_FILE = args.embeddingfile
    cnn_model = args.modelfile
    dictfile = args.dictfile
    maxlen = int(args.maxlen)
    
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_val = path['dir_val']

    word2vec_model = gensim.models.Word2Vec.load(dir_w2v + W2VEC_MODEL_FILE)
    model = load_model(dir_w2v + cnn_model)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    vocab = np.load(dir_w2v + dictfile).item()
    tp = TextProcessor()
    doc_list, tw_class = load_files(dir_val)
    tweets = tp.text_process(doc_list, text_only=True)
    tweets, tw_class = select_tweets(tweets, tw_class)

    x, y = gen_sequence()
    data = pad_sequences(x, maxlen=maxlen)
    y = np.array(y)

    y_pred = model.predict_on_batch(data)
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y, y_pred))

#python validation.py -f model_word2vec -m model_cnn.h5 -d dict_cnn.npy -l 18
#python validation.py -f model_word2vec -m model_lstm.h5 -d dict_lstm.npy -l 18
#python validation.py -f model_word2vec -m model_fast_text.h5 -d dict_fast_text.npy -l 18