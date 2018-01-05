import sys
sys.path.append('../')
import argparse
import configparser
import gensim
import json
import math
import numpy as np
import operator
import os
import pymongo
import sklearn
from batch_gen import batch_gen
from collections import defaultdict
from f_map import F_map
from gensim.parsing.preprocessing import STOPWORDS
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import Input
from keras.layers import Merge
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}
tw_class = list()


EMBEDDING_DIM = None
W2VEC_MODEL_FILE = None
SEED = 42
NO_OF_FOLDS = 10
CLASS_WEIGHT = None
LOSS_FUN = None
OPTIMIZER = None
KERNEL = None
MAX_SEQUENCE_LENGTH = None
INITIALIZE_WEIGHTS_WITH = None
LEARN_EMBEDDINGS = None
EPOCHS = 10
BATCH_SIZE = 512
SCALE_LOSS_FUN = None
MODEL_NAME = 'fast_text_model'
DICT_NAME = 'fast_text_dict'
DISPERSION = 'random'
SAMPLE = 2000

word2vec_model = None


def load_files(dir_in):
    doc_list = list()
    tw_files = sorted([file for root, dirs, files in os.walk(dir_in)
                 for file in files if file.endswith('.json')])
    tw_class = list()
    print(tw_files)
    for tw_file in tw_files:
        temp = list()
        with open(dir_in+tw_file) as data_file:
            for line in data_file:
                tweet = json.loads(line)
                temp.append(tweet['text'])
                doc_list.append(tweet['text'])
                tw_class.append(tw_file.split(".")[0])
    return doc_list, tw_class


def get_embedding(word):
    # return
    try:
        return word2vec_model[word]
    except e:
        print('Encoding not found: %s' % (word))
        return np.zeros(EMBEDDING_DIM)


def get_embedding_weights():
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model[k]
        except:
            n += 1
            pass
    print("%d embedding missed" % n)
    print("%d embedding found" % len(embedding))
    return embedding



def select_tweets(tweets, tw_class):
    # selects the tweets as in mean_glove_embedding method
    # Processing
    tweet_return = []
    class_return = []
    for i, tweet in enumerate(tweets):
        _emb = 0
        for w in tweet:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb += 1
        if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
            class_return.append(tw_class[i])
    print('Tweets selected:', len(tweet_return))
    return tweet_return, class_return

def gen_vocab(model_vec):
    vocab = dict([(k, v.index) for k, v in model_vec.vocab.items()])
    vocab['UNK'] = len(vocab) + 1
    print(vocab['UNK'])
    return vocab

def gen_sequence(vocab):
    y_map = {'politics': 0, 'non_politics': 1}
    X, y = [], []
    for i, tweet in enumerate(tweets):
        seq = []
        for word in tweet:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(y_map[tw_class[i]])
    return X, y

def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)


def fast_text_model(sequence_length):
    model = Sequential()
    model.add(Embedding(len(vocab)+1, EMBEDDING_DIM, input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    model.add(Dropout(0.5))
    model.add(GlobalAveragePooling1D())
    #model.add(Dense(len(set(tw_class)), activation='softmax'))
    model.add(Dense(len(set(tw_class)), activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model

def train_fast_text(X, y, model, inp_dim, weights, epochs=EPOCHS,
               batch_size=BATCH_SIZE):
    cv_object = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    print(cv_object)
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X.shape[1]
    lookup_table = np.zeros_like(model.layers[0].get_weights()[0])
    for train_index, test_index in cv_object.split(X):
        if INITIALIZE_WEIGHTS_WITH == "word2vec":
            model.layers[0].set_weights([weights])
        elif INITIALIZE_WEIGHTS_WITH == "random":
            shuffle_weights(model)
        else:
            print("ERROR!")
            return
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        for epoch in range(epochs):
            for X_batch in batch_gen(X_temp, batch_size):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]

                class_weights = None
                if SCALE_LOSS_FUN:
                    class_weights = {}
                    for cw in range(len(set(tw_class))):
                        class_weights[cw] = np.where(y_temp == cw)[0].shape[
                            0]/float(len(y_temp))
                try:
                    y_temp = np_utils.to_categorical(
                        y_temp, num_classes=len(set(tw_class)))
                except Exception as e:
                    print(e)
                    print(y_temp)
                loss, acc = model.train_on_batch(
                    x, y_temp, class_weight=class_weights)

        lookup_table += model.layers[0].get_weights()[0]
        y_pred = model.predict_on_batch(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        print(classification_report(y_test, y_pred))
        print(precision_recall_fscore_support(y_test, y_pred))
        print(y_pred)
        p += precision_score(y_test, y_pred, average='weighted')
        p1 += precision_score(y_test, y_pred, average='micro')
        r += recall_score(y_test, y_pred, average='weighted')
        r1 += recall_score(y_test, y_pred, average='micro')
        f1 += f1_score(y_test, y_pred, average='weighted')
        f11 += f1_score(y_test, y_pred, average='micro')

    print("macro results are")
    print("average precision is %f" % (p/NO_OF_FOLDS))
    print("average recall is %f" % (r/NO_OF_FOLDS))
    print("average f1 is %f" % (f1/NO_OF_FOLDS))

    print("micro results are")
    print("average precision is %f" % (p1/NO_OF_FOLDS))
    print("average recall is %f" % (r1/NO_OF_FOLDS))
    print("average f1 is %f" % (f11/NO_OF_FOLDS))
    return ((p / NO_OF_FOLDS), (r / NO_OF_FOLDS), (f1 / NO_OF_FOLDS))

def get_tweets(db, sample, dimension):
    sample = math.floor(sample / 2)
    tweets = list()
    tw_class = list()

    if dimension == 'few_months':
        tmp = db.politics.find().sort('created_at', pymongo.ASCENDING).limit(sample)
        for tw in tmp:
            tweets.append(tw['text_processed'].split(' '))
            tw_class.append('politics')
        tmp = db.non_politics.find().sort('created_at', pymongo.ASCENDING).limit(sample)
        for tw in tmp:
            tweets.append(tw['text_processed'].split(' '))
            tw_class.append('non_politics')

    elif dimension == 'few_parls':
        tmp = db.politics.aggregate(
            [
                {'$group': {'_id': "$user_id", 'text': {
                    '$push': "$text_processed"}, 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}}
            ]
        )
        x = 0
        for tw in tmp:
            if x <= sample:
                tweets += tw['text'][:(sample - x)]
                x += tw['count']
        tw_class += ['politics'] * len(tweets)
        tmp = db.non_politics.aggregate(
            [
                {'$group': {'_id': "$user_id", 'text': {
                    '$push': "$text_processed"}, 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}}
            ]
        )
        x = 0
        bf = len(tweets)
        for tw in tmp:
            if x <= sample:
                tweets += tw['text'][:(sample - x)]
                x += tw['count']
        tw_class += ['non_politics'] * (len(tweets) - bf)
        print('tamnho tw_class: %i' % len(tw_class))
        tweets = [t.split(' ') for t in tweets]

    else:
        tmp = db.politics.aggregate([{'$sample': {'size': sample}}])
        for tw in tmp:
            tweets.append(tw['text_processed'].split(' '))
            tw_class.append('politics')

        tmp = db.non_politics.aggregate([{'$sample': {'size': sample}}])
        for tw in tmp:
            tweets.append(tw['text_processed'].split(' '))
            tw_class.append('non_politics')

    return tweets, tw_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='LSTM based models for politics twitter')
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('--epochs', default=EPOCHS, required=True)
    parser.add_argument('--batch-size', default=BATCH_SIZE, required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)
    parser.add_argument('--initialize-weights',
                        choices=['random', 'word2vec'], required=True)
    parser.add_argument('--learn-embeddings',
                        action='store_true', default=False)
    parser.add_argument('--model_name', default=MODEL_NAME, required=True)
    parser.add_argument('--dict_name', default=DICT_NAME, required=True)
    parser.add_argument('--dispersion', default=DISPERSION, required=True)
    parser.add_argument('--sample', default=SAMPLE, required=True)
    parser.add_argument('--loss', default=LOSS_FUN, required=True)
       

    args = parser.parse_args()
    W2VEC_MODEL_FILE = args.embeddingfile
    EMBEDDING_DIM = int(args.dimension)
    SEED = int(args.seed)
    NO_OF_FOLDS = int(args.folds)
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    INITIALIZE_WEIGHTS_WITH = args.initialize_weights
    MODEL_NAME = args.model_name
    DICT_NAME = args.dict_name
    DISPERSION = args.dispersion
    SAMPLE = int(args.sample)
    LOSS_FUN = args.loss
    
    np.random.seed(SEED)
    print('W2VEC embedding: %s' % (W2VEC_MODEL_FILE))
    print('Embedding Dimension: %d' % (EMBEDDING_DIM))
    print('Allowing embedding learning: %s' % (str(LEARN_EMBEDDINGS)))

    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_in = path['dir_in']

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v + W2VEC_MODEL_FILE,
                                                                     binary=False,
                                                                     unicode_errors="ignore")

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    tweets, tw_class = get_tweets(db, SAMPLE, DISPERSION)
    tweets, tw_class = select_tweets(tweets, tw_class)

    vocab = gen_vocab(word2vec_model)
    X, y = gen_sequence(vocab)
    MAX_SEQUENCE_LENGTH = max(map(lambda x: len(x), X))
    print("max seq length is %d" % (MAX_SEQUENCE_LENGTH))

    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    W = get_embedding_weights()

    model = fast_text_model(data.shape[1])
    p, r, f1 = train_fast_text(data, y, model, EMBEDDING_DIM, W)
    model.save(dir_w2v + MODEL_NAME + ".h5")
    np.save(dir_w2v + DICT_NAME + '.npy', vocab)
    txt = '%i, %i, %i, %i, %i, %.2f, %.2f, %.2f, ' % (F_map.get_id('FAST_TEXT'), F_map.get_id(W2VEC_MODEL_FILE),
                                   F_map.get_id(EMBEDDING_DIM), F_map.get_id(SAMPLE), F_map.get_id(DISPERSION),
                                   p, r, f1)
    f = open(dir_w2v + "trainned_params.txt", 'a')
    f.write(txt)
    f.close()

    # python fast_text.py -f model_word2vec -d 100 --initialize-weights word2vec --learn-embeddings --epochs 10 --batch-size 30
    
