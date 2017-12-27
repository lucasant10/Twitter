import sys
sys.path.append('../')
import argparse
import configparser
import gensim
import sklearn
import h5py
import json
import math
import numpy as np
import os
import os
import pymongo
from batch_gen import batch_gen
from collections import defaultdict
from f_map import F_map
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Convolution1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import MaxPooling1D
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
from political_classification import PoliticalClassification
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}


EMBEDDING_DIM = None
W2VEC_MODEL_FILE = None
NO_OF_CLASSES = 2
MAX_SEQUENCE_LENGTH = 18
SEED = 42
NO_OF_FOLDS = 10
CLASS_WEIGHT = None
LOSS_FUN = None
OPTIMIZER = None
TOKENIZER = None
INITIALIZE_WEIGHTS_WITH = None
LEARN_EMBEDDINGS = None
EPOCHS = 10
BATCH_SIZE = 30
SCALE_LOSS_FUN = None
MODEL_NAME = 'cnn_model'
DICT_NAME = 'cnn_dict'
DISPERSION = 'random'
SAMPLE = 2000

word2vec_model = None


def get_embedding_weights(vocab):
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


def gen_vocab(model_vec):
    vocab = dict([(k, v.index) for k, v in model_vec.vocab.items()])
    vocab['UNK'] = len(vocab) + 1
    print(vocab['UNK'])
    return vocab


def gen_sequence(vocab, tweets):
    X, y= [],[]
    for tweet in tweets:
        seq = []
        for word in tweet[0]:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(tweet[1])
    print(y)
    return X, y


def cnn_model(sequence_length, embedding_dim):
    model_variation = 'CNN-rand'  # CNN-rand | CNN-non-static | CNN-static
    print('Model variation is %s' % model_variation)

    # Model Hyperparameters
    n_classes = NO_OF_CLASSES
    embedding_dim = EMBEDDING_DIM
    filter_sizes = (3, 4, 5)
    num_filters = 120
    dropout_prob = (0.25, 0.25)
    hidden_dims = 100

    # Training parameters
    # Word2Vec parameters, see train_word2vec
    # min_word_count = 1  # Minimum word count
    # context = 10        # Context window size

    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu')(graph_in)
        #,subsample_length=1)(graph_in)
        pool = GlobalMaxPooling1D()(conv)
        #flatten = Flatten()(pool)
        convs.append(pool)

    if len(filter_sizes) > 1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    # main sequential model
    model = Sequential()
    # if not model_variation=='CNN-rand':
    model.add(Embedding(len(vocab)+1, embedding_dim,
                        input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    # , input_shape=(sequence_length, embedding_dim)))
    model.add(Dropout(dropout_prob[0]))
    model.add(graph)
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    # model.add(Dense(n_classes))
    # model.add(Activation('softmax'))
    #model.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model

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
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model


def train_CNN(X, y, inp_dim, model, weights, epochs=EPOCHS, batch_size=BATCH_SIZE):
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
                    for cw in range(2):
                        class_weights[cw] = np.where(y_temp == cw)[0].shape[
                            0]/float(len(y_temp))
                try:
                    y_temp = np_utils.to_categorical(
                        y_temp, num_classes = 2)
                except Exception as e:
                    print(e)
                    print(y_temp)
                #print(x.shape, y.shape)
                loss, acc = model.train_on_batch(
                    x, y_temp, class_weight=class_weights)

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
    print("average precision is %f" % (p / NO_OF_FOLDS))
    print("average recall is %f" % (r / NO_OF_FOLDS))
    print("average f1 is %f" % (f1 / NO_OF_FOLDS))

    print("micro results are")
    print("average precision is %f" % (p1 / NO_OF_FOLDS))
    print("average recall is %f" % (r1 / NO_OF_FOLDS))
    print("average f1 is %f" % (f11 / NO_OF_FOLDS))

    return ((p / NO_OF_FOLDS), (r / NO_OF_FOLDS), (f1 / NO_OF_FOLDS))

def learn_predict(vocab, tweets):
    X, y = gen_sequence(vocab, tweets)
    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    model = lstm_model(data.shape[1], EMBEDDING_DIM)
    p, r, f1 = train_CNN(data, y, EMBEDDING_DIM, model, W)
    return (p, r, f1)


def save(cond, label, p, r, f1):
    print('saving !!')
    txt = '%s, %i, %.2f, %.2f, %.2f \n' % (
        F_map.get_id(cond), F_map.get_id(label), p, r, f1)
    f = open(dir_w2v + "predict_politics.txt", 'a')
    f.write(txt)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CNN based models for predict politics')
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('--epochs', default=EPOCHS, required=True)
    parser.add_argument('--batch-size', default=BATCH_SIZE, required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)
    parser.add_argument('--class_weight')
    parser.add_argument('--initialize-weights',
                        choices=['random', 'word2vec'], required=True)
    parser.add_argument('--learn-embeddings',
                        action='store_true', default=False)
    args = parser.parse_args()

    W2VEC_MODEL_FILE = args.embeddingfile
    EMBEDDING_DIM = int(args.dimension)
    SEED = int(args.seed)
    NO_OF_FOLDS = int(args.folds)
    CLASS_WEIGHT = args.class_weight
    INITIALIZE_WEIGHTS_WITH = args.initialize_weights
    LEARN_EMBEDDINGS = args.learn_embeddings
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)

    np.random.seed(SEED)
    print('W2VEC embedding: %s' % (W2VEC_MODEL_FILE))
    print('Embedding Dimension: %d' % (EMBEDDING_DIM))
    print('Allowing embedding learning: %s' % (str(LEARN_EMBEDDINGS)))

    # election
    p1 = (1396483200000, 1443830400000)
    p2 = (1443830400000, 1428019200000)
    # impeachment
    p3 = (1427760000000, 1472601600000)
    p4 = (1472601600000, 1490918400000)

    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_in = path['dir_in']

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb

    pc = PoliticalClassification('model_lstm.h5',
                                 'dict_lstm.npy', 16)

    vocab = pc.vocab
    W = get_embedding_weights(vocab)
    conditions = [('novos', 'reeleitos'), ('reeleitos','nao_eleitos'), ('novos', 'nao_eleitos')]
    cond_map = {'novos': 0, 'reeleitos': 1, 'nao_eleitos': 2}
    periods = [p1, p2, p3, p4]
    for period in periods:
        # tweets = db.tweets.find({'created_at': {'$gte': period[0], '$lt': period[1]},
        #                          'cond_55': {'$exists': True}}
        tweets = db.tweets.aggregate([{'$sample': {'size': 3000}},
                                      {'$match': {'created_at': {'$gte': period[0], '$lt': period[1]},
                                                  'cond_55': {'$exists': True}}}])

        politics = defaultdict(list)
        non_politics = defaultdict(list)
        print('getting data')
        for tweet in tweets:
            if pc.is_political(tweet['text_processed']):
                politics[tweet['cond_55']].append((tweet['text_processed'].split(' '), cond_map[tweet['cond_55']]))
            else:
                non_politics[tweet['cond_55']].append((tweet['text_processed'].split(' '), cond_map[tweet['cond_55']] ))

        for con in conditions:
            print('processing politics')
            tweets_p = politics[con[0]] + politics[con[1]]
            p, r, f1 = learn_predict(vocab, tweets_p)
            save(con, 'politics', p, r, f1)

            print('processing Non politics')
            tweets_n_p = non_politics[con[0]] + non_politics[con[1]]
            p, r, f1 = learn_predict(vocab, tweets_n_p)
            save(con, 'non_politics', p, r, f1)

            print('processing all')
            tweets_all = tweets_p + tweets_n_p
            p, r, f1 = learn_predict(vocab, tweets_all)
            save(con, 'all', p, r, f1)

# python predict_politics.py -f cbow_s100.txt  -d 100 --epochs 10
# --batch-size 30 --initialize-weights word2vec --learn-embeddings
