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
from keras import backend as K
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
from gensim.models import Doc2Vec
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
MODEL_TYPE = 'logistic'

word2vec_model = None

def gen_sequence(deps, con):
    y_map = {con[0]: 0, con[1]: 1}
    X, y= [],[]
    for tws in deps:
        d_class = tws[0].split('-')[1]
        doc = ' '.join(tws[1])
        X.append(doc2vec_model.infer_vector(doc))
        y.append(y_map[d_class])
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

def baseline_model(embedding_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=embedding_dim, init='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, init='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, init='normal', activation='relu'))  
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model


def train_CNN(X, y, inp_dim, model,  epochs=EPOCHS, batch_size=BATCH_SIZE):
    cv_object = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    print(cv_object)
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X.shape[1]
    for train_index, test_index in cv_object.split(X):
        # if INITIALIZE_WEIGHTS_WITH == "word2vec":
        #     model.layers[0].set_weights([weights])
        # elif INITIALIZE_WEIGHTS_WITH == "random":
        #     shuffle_weights(model)
        # else:
        #     print("ERROR!")
        #     return
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

    del model
    K.clear_session()

    return ((p / NO_OF_FOLDS), (r / NO_OF_FOLDS), (f1 / NO_OF_FOLDS))

def train_svm(X, Y):
    model = classification_model(X, Y, MODEL_TYPE)
    result = model.predict(X)
    print(classification_report(Y, result))
    
def learn_predict(tweets, con):
    X, y = gen_sequence(tweets,con)
    y = np.array(y)
    X = np.array(X)
    data, y = sklearn.utils.shuffle(X, y)
    #model = baseline_model(EMBEDDING_DIM)
    #p, r, f1 = train_CNN(data, y, EMBEDDING_DIM, model)
    p, r, f1 = classification_model(data, y, MODEL_TYPE)
    return (p, r, f1)


def save(period, cond, label, p, r, f1):
    print('saving !!')
    txt = '%i, %s, %i, %.2f, %.2f, %.2f \n' % (
        F_map.get_id(period), F_map.get_id(cond), F_map.get_id(label), p, r, f1)
    f = open(dir_w2v + "predict_politics.txt", 'a')
    f.write(txt)
    f.close()

def classification_model(X, Y, model_type=None):
    X, Y = shuffle(X, Y, random_state=SEED)
    print("Model Type:", model_type)

    #predictions = cross_val_predict(logreg, X, Y, cv=NO_OF_FOLDS)
    model = get_model(model_type)
    scores1 = cross_val_score(model.fit(X,Y), X, Y, cv=NO_OF_FOLDS, scoring='precision_weighted')
    print("Precision(avg): %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std() * 2))

    scores2 = cross_val_score(get_model(model_type), X, Y, cv=NO_OF_FOLDS, scoring='recall_weighted')
    print("Recall(avg): %0.3f (+/- %0.3f)" % (scores2.mean(), scores2.std() * 2))
    
    scores3 = cross_val_score(get_model(model_type), X, Y, cv=NO_OF_FOLDS, scoring='f1_weighted')
    print("F1-score(avg): %0.3f (+/- %0.3f)" % (scores3.mean(), scores3.std() * 2))

    return (scores1.mean(), scores2.mean(), scores3.mean())

def get_model(m_type=None):
    if not m_type:
        print("ERROR: Please specify a model type!")
        return None
    if m_type == 'logistic':
        logreg = LogisticRegression()
    elif m_type == "gradient_boosting":
        logreg = GradientBoostingClassifier(loss=LOSS_FUN, n_estimators=N_ESTIMATORS)
    elif m_type == "random_forest":
        logreg = RandomForestClassifier(class_weight=CLASS_WEIGHT, n_estimators=N_ESTIMATORS)
    elif m_type == "svm":
        logreg = SVC(class_weight=CLASS_WEIGHT, kernel=KERNEL)
    elif m_type == "svm_linear":
        logreg = LinearSVC(loss=LOSS_FUN, class_weight=CLASS_WEIGHT)
    else:
        print("ERROR: Please specify a correct model")
        return None
    return logreg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CNN based models for predict politic class')
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
    p1 = (1396483200000, 1412294400000)
    p2 = (1412294400000, 1443830400000)
    # impeachment
    p3 = (1459382400000, 1472601600000)
    p4 = (1472601600000, 1490918400000)

    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_in = path['dir_in']

    doc2vec_model = Doc2Vec.load(dir_w2v + "model_doc2vec.d2v")

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb

    pc = PoliticalClassification('model_lstm.h5',
                                 'dict_lstm.npy', 16)

    # vocab = pc.vocab
    # W = get_embedding_weights(vocab)
    conditions = [('novos', 'reeleitos'), ('reeleitos','nao_eleitos'), ('novos', 'nao_eleitos')]
    cond_map = {'novos': 0, 'reeleitos': 1, 'nao_eleitos': 2}
    periods = [p1]
    for period in periods:
        #tweets = db.tweets.find({'created_at': {'$gte': period[0], '$lt': period[1]},
        #                          'cond_55': {'$exists': True}})
        print('getting data')
        tw = db.tweets.aggregate([ { '$sample': { 'size': 10000 }}, 
                                      {'$match': {'created_at': {'$gte': period[0], '$lt': period[1]},
                                                  'cond_55': {'$exists': True}}}], allowDiskUse=True)

        politics = dict({'nao_eleitos': defaultdict(list), 'reeleitos': defaultdict(list), 'novos': defaultdict(list)})
        non_politics = dict({'nao_eleitos': defaultdict(list), 'reeleitos': defaultdict(list), 'novos': defaultdict(list)})
        both = dict({'nao_eleitos': defaultdict(list), 'reeleitos': defaultdict(list), 'novos': defaultdict(list)})
        print('separating tweets')
        for tweet in tw:
            if pc.is_political(tweet['text_processed']):
                politics[tweet['cond_55']][str(tweet['user_id'] + '-' + tweet['cond_55']) ].append((tweet['text_processed']))
            else:
                non_politics[tweet['cond_55']][str(tweet['user_id'] + '-' + tweet['cond_55'])].append((tweet['text_processed']))
            both[tweet['cond_55']][str(tweet['user_id'] + '-' + tweet['cond_55'])].append(tweet['text_processed'])

        for con in conditions:
            print('processing politics')
            tweets_p = list()
            for dep in politics[con[0]].items():
                tweets_p.append(dep)
            for dep in politics[con[1]].items():
                tweets_p.append(dep)
            p, r, f1 = learn_predict(tweets_p, con)
            save(period[0], con, 'politics', p, r, f1)

            print('processing Non politics')
            tweets_n_p = list()
            for dep in non_politics[con[0]].items():
                tweets_n_p.append(dep)
            for dep in non_politics[con[1]].items():
                tweets_n_p.append(dep)
            p, r, f1 = learn_predict(tweets_n_p, con)
            save(period[0], con, 'non_politics', p, r, f1)

            print('processing all')
            tweets_b = list()
            for dep in both[con[0]].items():
                tweets_b.append(dep)
            for dep in both[con[1]].items():
                tweets_b.append(dep)
            p, r, f1 = learn_predict(tweets_b, con)
            save(period[0], con, 'all', p, r, f1)

# python class_prediction.py -f model_doc2vec.txt  -d 300 --epochs 10 --batch-size 30 --initialize-weights word2vec
