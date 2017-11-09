import sys
sys.path.append('../')
import argparse
import configparser
import numpy as np
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.utils import shuffle
import gensim, sklearn
from collections import defaultdict
from batch_gen import batch_gen
from text_processor import TextProcessor
import json
import os


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



def gen_data(tweets, tw_class):
    y_map = dict()
    for i, v in enumerate(sorted(set(tw_class))):
        y_map[v] = i
    print(y_map)

    X, y = [], []
    for i, tweet in enumerate(tweets):
        emb = np.zeros(EMBEDDING_DIM)
        for word in tweet:
            try:
                emb += word2vec_model[word]
            except:
                pass
        emb /= len(tweet)
        X.append(emb)
        y.append(y_map[tw_class[i]])
    return X, y

def select_tweets(tweets):
    # selects the tweets as in mean_glove_embedding method
    # Processing       
    X, Y = [], []
    tweet_return = []
    for tweet in tweets:
        _emb = 0
        for w in tweet:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb += 1
        if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
    print('Tweets selected:', len(tweet_return))
    return tweet_return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BagOfWords model validation')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    

    args = parser.parse_args()
    MODEL_FILE = args.model
    W2VEC_MODEL_FILE = args.embeddingfile
    EMBEDDING_DIM = int(args.dimension)
    
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_val = path['dir_val']

    word2vec_model = gensim.models.Word2Vec.load(dir_w2v+W2VEC_MODEL_FILE)
    tp = TextProcessor()
    doc_list, tw_class = load_files(dir_val)
    tweets = tp.text_process(doc_list, text_only=True)
    tweets = select_tweets(tweets)

    X, Y = gen_data(tweets, tw_class)

    model = joblib.load(dir_w2v + MODEL_FILE)
    result = model.predict(X)
    print(classification_report(Y, result))
    


#python validation_BoW.py -m logistic.skl -f model_word2vec -d 100