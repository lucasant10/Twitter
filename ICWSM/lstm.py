import sys
sys.path.append('../')
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Bidirectional
import numpy as np
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import KFold
from keras.utils import np_utils
import operator
import gensim
import sklearn
from collections import defaultdict
from batch_gen import batch_gen
import os
import configparser
from text_processor import TextProcessor
import json
import pymongo
import math

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
MODEL_NAME = 'lstm_model'
DICT_NAME = 'lstm_dict'
DISPERSION = 'uniform'
SAMPLE = 2000

word2vec_model = None


def load_files(dir_in):
    doc_list = list()
    tw_files = sorted([file for root, dirs, files in os.walk(dir_in)
                 for file in files if file.endswith('.json')])
    tw_class = list()
    for tw_file in tw_files:
        temp = list()
        with open(dir_in + tw_file) as data_file:
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


def gen_vocab(model_vec):
    vocab = dict([(k, v.index) for k, v in model_vec.vocab.items()])
    vocab['UNK'] = len(vocab) + 1
    print(vocab['UNK'])
    return vocab


def filter_vocab(k):
    global freq, vocab
    freq_sorted = sorted(freq.items(), key=operator.itemgetter(1))
    tokens = freq_sorted[:k]
    vocab = dict(zip(tokens, range(1, len(tokens) + 1)))
    vocab['UNK'] = len(vocab) + 1


def gen_sequence(vocab):
    y_map = dict()
    for i, v in enumerate(set(tw_class)):
        y_map[v] = i
    print(y_map)

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


def lstm_model(sequence_length, embedding_dim):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, embedding_dim,
                        input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    model.add(Dropout(0.25))
    model.add(LSTM(50))
    model.add(Dropout(0.5))
    # model.add(Dense(len(set(tw_class)),  activation= 'softmax' ))
    # model.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])
    model.add(Dense(len(set(tw_class)), activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model


def train_LSTM(X, y, model, inp_dim, weights, epochs=EPOCHS,
               batch_size=BATCH_SIZE):
    cv_object = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    print(cv_object)
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X.shape[1]
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
                            0] / float(len(y_temp))
                try:
                    y_temp = np_utils.to_categorical(
                        y_temp, num_classes=len(set(tw_class)))
                except Exception as e:
                    print(e)
                    print(y_temp)
                loss, acc = model.train_on_batch(
                    x, y_temp, class_weight=class_weights)
                # print("Loss: %d, Acc: %d"%(loss, acc))

        y_pred = model.predict_on_batch(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        print(classification_report(y_test, y_pred))
        print(precision_recall_fscore_support(y_test, y_pred))
        p += precision_score(y_test, y_pred, average='weighted')
        p1 += precision_score(y_test, y_pred, average='micro')
        r += recall_score(y_test, y_pred, average='weighted')
        r1 += recall_score(y_test, y_pred, average='micro')
        f1 += f1_score(y_test, y_pred, average='weighted')
        f11 += f1_score(y_test, y_pred, average='micro')

    print("macro results are")
    print("average precision is %f" % (p / NO_OF_FOLDS))
    print("average recall is %f" % (r / NO_OF_FOLDS))
    print("average f1 is %f" % (f1 / NO_OF_FOLDS))

    print("micro results are")
    print("average precision is %f" % (p1 / NO_OF_FOLDS))
    print("average recall is %f" % (r1 / NO_OF_FOLDS))
    print("average f1 is %f" % (f11 / NO_OF_FOLDS))

    txt = 'average precision \t average recall \t average F1'
    txt = ' %f \t  %f \t %f ' % (
        (p / NO_OF_FOLDS), (r / NO_OF_FOLDS), (f1 / NO_OF_FOLDS))
    return txt


def get_tweets(db, sample, dimension):
    sample = math.floor(sample / 2)
    tweets = list()
    if dimension == 'few_month':
        tweets = db.politics.find().sort('created_at', pymongo.ASCENDING).limit(sample)
        tweets += db.non_politics.find().sort('created_at', pymongo.ASCENDING).limit(sample)
    elif dimension == 'few_parl':
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
        tmp = db.non_politics.aggregate(
                [
                    {'$group': {'_id': "$user_id", 'text': {'$push': "$text_processed"}, 'count': {'$sum': 1}}},
                    {'$sort': {'count': -1}}
                ]
            )
        x = 0
        for tw in tmp:
            if x <= sample:
                tweets += tw['text'][:(sample - x)]
                x += tw['count']

    else:
        tmp = db.politics.aggregate([{ '$sample': { 'size': sample }}])
        for tw in tmp:
            tweets += tw['text_processed'].split(' ')
        tmp = db.non_politics.aggregate([{ '$sample': { 'size': sample }}])
        for tw in tmp:
            tweets += tw['text_processed'].split(' ')
    return tweets
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='LSTM based models for politics twitter')
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('--loss', default=LOSS_FUN, required=True)
    parser.add_argument('--epochs', default=EPOCHS, required=True)
    parser.add_argument('--batch-size', default=BATCH_SIZE, required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)
    parser.add_argument('--kernel', default=KERNEL)
    parser.add_argument('--class_weight')
    parser.add_argument('--initialize-weights',
                        choices=['random', 'word2vec'], required=True)
    parser.add_argument('--learn-embeddings',
                        action='store_true', default=False)
    parser.add_argument('--scale-loss-function',
                        action='store_true', default=False)
    parser.add_argument('--model_name', default=MODEL_NAME, required=True)
    parser.add_argument('--dict_name', default=DICT_NAME, required=True)
    parser.add_argument('--dispersion', default=DISPERSION, required=True)
    parser.add_argument('--sample', default=SAMPLE, required=True)

    args = parser.parse_args()
    W2VEC_MODEL_FILE = args.embeddingfile
    EMBEDDING_DIM = int(args.dimension)
    SEED = int(args.seed)
    NO_OF_FOLDS = int(args.folds)
    CLASS_WEIGHT = args.class_weight
    LOSS_FUN = args.loss
    KERNEL = args.kernel
    INITIALIZE_WEIGHTS_WITH = args.initialize_weights
    LEARN_EMBEDDINGS = args.learn_embeddings
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    SCALE_LOSS_FUN = args.scale_loss_function
    MODEL_NAME = args.model_name
    DICT_NAME = args.dict_name
    DISPERSION = args.dispersion
    SAMPLE = int(args.sample)

    np.random.seed(SEED)
    print('W2VEC embedding: %s' % (W2VEC_MODEL_FILE))
    print('Embedding Dimension: %d' % (EMBEDDING_DIM))
    print('Allowing embedding learning: %s' % (str(LEARN_EMBEDDINGS)))

    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_in = path['dir_in']

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v+W2VEC_MODEL_FILE,
                                                   binary=False,
                                                   unicode_errors="ignore")

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    tweets = get_tweets(db, SAMPLE, DISPERSION)
    tweets = select_tweets(tweets)

    vocab = gen_vocab(word2vec_model)
    X, y = gen_sequence(vocab)
    MAX_SEQUENCE_LENGTH = max(map(lambda x: len(x), X))
    print("max seq length is %d" % (MAX_SEQUENCE_LENGTH))

    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    W = get_embedding_weights()

    model = lstm_model(data.shape[1], EMBEDDING_DIM)
    txt = train_LSTM(data, y, model, EMBEDDING_DIM, W)
    model.save(dir_w2v + MODEL_NAME + ".h5")
    np.save(dir_w2v + DICT_NAME + '.npy', vocab)
    f = open(dir_w2v + "trainned_params.txt", 'a+')
    f.write(txt)
    f.close()



    # lstm.py -f model_word2vec -d 100 --loss categorical_crossentropy --initialize-weights word2vec --learn-embeddings --epochs 10 --batch-size 30
    # lstm.py -f cbow_s100.txt -d 100 --loss categorical_crossentropy --initialize-weights word2vec --learn-embeddings --epochs 10 --batch-size 30
    

