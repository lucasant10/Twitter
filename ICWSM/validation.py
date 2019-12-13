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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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

def get_tweets(dfe):
    tw_class = list()
    political = list()
    npolitical = list()

    political = dfe[dfe.apply(lambda x: x['political'] == True, axis=1)]['text_processed'].tolist()
    tw_class = ['politics'] * len(political)

    npolitical = dfe[dfe.apply(lambda x: x['political'] == False, axis=1)]['text_processed'].tolist()
    tw_class += ['non_politics'] * len(npolitical)
    tweets = political + npolitical
    return tweets, tw_class
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validation of politics twitter model')
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-m', '--modelfile', required=True)
    parser.add_argument('-d', '--dictfile', required=True)
    parser.add_argument('-l', '--maxlen', required=True)

    
    args = parser.parse_args()

    W2VEC_MODEL_FILE = args.embeddingfile
    arg_model = args.modelfile
    dictfile = args.dictfile
    maxlen = int(args.maxlen)
    
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_val = path['dir_val']
    dir_data = path['dir_out']

    print('loading vector model')
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v + W2VEC_MODEL_FILE,
                                                   binary=False,
                                                   unicode_errors="ignore")

    df = pd.read_pickle(dir_data + 'validation/validation.pck')
    texts = list()
    tw_class = list()
    texts, tw_class = get_tweets(df)
    texts, tw_class = select_texts(texts, tw_class)
    
    print('load w2v model')
    model = load_model(dir_w2v + arg_model + '.h5')
    vocab = np.load(dir_w2v + dictfile + '.npy').item()

    X, y = gen_sequence(vocab, texts, tw_class)
    data = pad_sequences(X, maxlen=maxlen)
    y = np.array(y)
    print('predicting')
    y_pred = model.predict_on_batch(data)
    y_pred = np.argmax(y_pred, axis=1)
    p, r, f, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
    txt = '%.2f, %.2f, %.2f \n' % (p, r, f)
    f = open(dir_w2v + "trainned_params.txt", 'a')
    f.write(txt)
    f.close()


#python validation.py -f model_word2vec -m model_cnn.h5 -d dict_cnn.npy -l 18
#python validation.py -f model_word2vec -m model_lstm.h5 -d dict_lstm.npy -l 18
#python validation.py -f model_word2vec -m model_fast_text.h5 -d dict_fast_text.npy -l 18

#python validation.py -f cbow_s100.txt -m model_cnn.h5 -d dict_cnn.npy -l 18
#python validation.py -f cbow_s100.txt -m model_lstm.h5 -d dict_lstm.npy -l 18