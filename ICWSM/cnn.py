# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout,concatenate, Flatten, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
import numpy as np
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
from keras.utils import np_utils
import gensim, sklearn
from collections import defaultdict
from batch_gen import batch_gen
import os
import configparser
import json
import h5py
import math
from f_map import F_map
import os
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}



EMBEDDING_DIM = None
W2VEC_MODEL_FILE = None
NO_OF_CLASSES=2
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


def select_texts(texts, classes):
    # selects the texts as in embedding method
    # Processing
    text_return = []
    class_return = []
    for i, text in enumerate(texts):
        _emb = 0
        for w in text:
            if w in word2vec_model:  # Check if embeeding there in embedding model
                _emb += 1
        if _emb:   # Not a blank text
            text_return.append(text)
            class_return.append(classes[i])
    print('texts selected:', len(text_return))
    return text_return, class_return

def gen_vocab(model_vec):
    vocab = dict([(k, v.index) for k, v in model_vec.vocab.items()])
    vocab['UNK'] = len(vocab) + 1
    print(vocab['UNK'])
    return vocab

def gen_sequence(vocab, texts, tw_class):
    y_map = dict()
    for i, v in enumerate(sorted(set(tw_class))):
        y_map[v] = i
    print(y_map)
    X, y = [], []
    for i, text in enumerate(texts):
        seq = []
        for word in text:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(y_map[tw_class[i]])
    return X, y

def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

def cnn_model(sequence_length, embedding_dim):
    model_variation = 'CNN-rand'  #  CNN-rand | CNN-non-static | CNN-static
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
    #min_word_count = 1  # Minimum word count
    #context = 10        # Context window size

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

    if len(filter_sizes)>1:
        out = concatenate(convs)
        #out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    # main sequential model
    model = Sequential()
    #if not model_variation=='CNN-rand':
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    model.add(Dropout(dropout_prob[0]))#, input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    #model.add(Dense(n_classes))
    #model.add(Activation('softmax'))
    #model.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])
    model.add(Dense(len(set(tw_class)), activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
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
                    for cw in range(len(set(tw_class))):
                        class_weights[cw] = np.where(y_temp == cw)[0].shape[
                            0]/float(len(y_temp))
                try:
                    y_temp = np_utils.to_categorical(
                        y_temp, num_classes=len(set(tw_class)))
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

def get_tweets(dfe, sample, dimension):
    sample = math.floor(sample / 2)
    tw_class = list()
    political = list()
    npolitical = list()

    if dimension == 'few_months':

        political = dfe[dfe.apply(lambda x: x['political'] 
                == True, axis=1)].sort_values(by=['created_at'],ascending=False)[:sample]['text_processed'].tolist()
        tw_class = ['politics'] * len(political)

        npolitical = dfe[dfe.apply(
                lambda x: x['political'] == False, axis=1)].sort_values(by=['created_at'],ascending=False)[:sample]['text_processed'].tolist()
        tw_class += ['non_politics'] * len(npolitical)

    elif dimension == 'few_parls':

        political = dfe[dfe.apply(lambda x: x['political'] == True, axis=1)].groupby(['user_id','text_processed']).size().to_frame('count').reset_index().sort_values(['count'])[:sample]['text_processed'].tolist()
        tw_class = ['politics'] * len(political)
        npolitical = dfe[dfe.apply(lambda x: x['political'] == False, axis=1)].groupby(['user_id','text_processed']).size().to_frame('count').reset_index().sort_values(['count'])[:sample]['text_processed'].tolist()
        tw_class += ['non_politics'] * len(npolitical)

    else:

        political = dfe[dfe.apply(lambda x: x['political'] == True, axis=1)].sample(n=sample, random_state=1)['text_processed'].tolist()
        tw_class = ['politics'] * len(political)

        npolitical = dfe[dfe.apply(lambda x: x['political'] == False, axis=1)].sample(n=sample, random_state=1)['text_processed'].tolist()
        tw_class += ['non_politics'] * len(npolitical)
    tweets = political + npolitical
    return tweets, tw_class
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN based models for politics twitter')
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('--loss', default=LOSS_FUN, required=True)
    parser.add_argument('--epochs', default=EPOCHS, required=True)
    parser.add_argument('--batch-size', default=BATCH_SIZE, required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)
    parser.add_argument('--class_weight')
    parser.add_argument('--initialize-weights', choices=['random', 'word2vec'], required=True)
    parser.add_argument('--learn-embeddings', action='store_true', default=False)
    parser.add_argument('--scale-loss-function', action='store_true', default=False)
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
    dir_data = path['dir_out']

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v+W2VEC_MODEL_FILE,
                                                   binary=False,
                                                   unicode_errors="ignore")

    df = pd.read_pickle(dir_data + 'trainning/trainning.pck')
    texts = list()
    tw_class = list()
    texts, tw_class = get_tweets(df, SAMPLE, DISPERSION)
    texts, tw_class = select_texts(texts, tw_class)
    print(tw_class)
    
    vocab = gen_vocab(word2vec_model)
    X, y = gen_sequence(vocab, texts, tw_class)

    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    W = get_embedding_weights()
    model = cnn_model(data.shape[1], EMBEDDING_DIM)
    p, r, f1 = train_CNN(data, y, EMBEDDING_DIM, model, W)
    model.save(dir_w2v + MODEL_NAME + ".h5")
    np.save(dir_w2v + DICT_NAME + '.npy', vocab)
    txt = '%i, %i, %i, %i, %i, %.2f, %.2f, %.2f, ' % (F_map.get_id('CNN'), F_map.get_id(W2VEC_MODEL_FILE),
                                   F_map.get_id(EMBEDDING_DIM), F_map.get_id(SAMPLE), F_map.get_id(DISPERSION),
                                   p, r, f1)
    f = open(dir_w2v + "trainned_params.txt", 'a')
    f.write(txt)
    f.close()


#python cnn.py -f model_word2vec -d 50 --loss categorical_crossentropy --optimizer adam --epochs 10 --batch-size 30 --initialize-weights word2vec --scale-loss-function
#python cnn.py -f model_word2vec -d 100 --loss categorical_crossentropy --optimizer adam --epochs 10 --batch-size 30 --initialize-weights word2vec --learn-embeddings
#python cnn.py -f cbow_s100.txt  -d 100 --loss categorical_crossentropy --optimizer adam --epochs 10 --batch-size 30 --initialize-weights word2vec --learn-embeddings