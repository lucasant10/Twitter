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
import pymongo
from political_classification import PoliticalClassification

def select_tweets(tweets):
    # selects the tweets as in mean_glove_embedding method
    # Processing       
    X = []
    tweet_return = []
    for i, tweet in enumerate(tweets):
        _emb = 0
        for w in tweet:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb += 1
        if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
    print('Tweets selected:', len(tweet_return))
    return tweet_return

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
    X = []
    for i, tweet in enumerate(tweets):
        seq = []
        for word in tweet:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
    return X



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN based models for politics twitter')
    parser.add_argument('-m', '--modelfile', required=True)
    parser.add_argument('-d', '--dictfile', required=True)
    parser.add_argument('-l', '--maxlen', required=True)
    parser.add_argument('--db',
                        action='store_true', default=False)

    
    args = parser.parse_args()
    arg_model = args.modelfile
    dictfile = args.dictfile
    maxlen = int(args.maxlen)
    database = args.db
    
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_ale = path['dir_ale']

    pc = PoliticalClassification(arg_model, dictfile, maxlen)

    txt = ''
    if database:
        print('loading from mongodb')
        client = pymongo.MongoClient("mongodb://localhost:27017")
        db = client.twitterdb
        tweets = db.tweets.find({'created_at': {'$gte': 1380585600000, '$lt': 1506816000000}})
        for tw in tweets:
            if pc.is_political(tw['text_processed']):
                txt += tw['text_processed'] + '\n'
    else:    
        tp = TextProcessor()
        print('loading from files')
        doc_list, tw_class = load_files(dir_ale)
        tmp = tp.text_process(doc_list, text_only=True)
        for tw in tmp:
            if pc.is_political(' '.join(tw)):
                txt += ' '.join(tw) + '\n'
                print(' '.join(tw))
    f = open(dir_ale + "politics_text.txt", 'w')
    f.write(txt)
    f.close()


#python select_politics.py -f cbow_s100.txt -m model_lstm.h5 -d dict_lstm.npy -l 18 --db
