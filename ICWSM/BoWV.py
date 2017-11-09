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



### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids


# logistic, gradient_boosting, random_forest, svm_linear, svm_rbf
W2VEC_MODEL_FILE = None
EMBEDDING_DIM = None
MODEL_TYPE = None
CLASS_WEIGHT = None
N_ESTIMATORS = 10
LOSS_FUN = 'deviance'
KERNEL = 'rbf'
TOKENIZER = None

SEED=42
MAX_NB_WORDS = None
NO_OF_FOLDS=10


# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}


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

    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BagOfWords model for politics twitter')
    parser.add_argument('-m', '--model', choices=['logistic', 'gradient_boosting', 'random_forest', 'svm', 'svm_linear'], required=True)
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)
    parser.add_argument('--estimators', default=N_ESTIMATORS)
    parser.add_argument('--loss', default=LOSS_FUN)
    parser.add_argument('--kernel', default=KERNEL)
    parser.add_argument('--class_weight')


    args = parser.parse_args()
    MODEL_TYPE = args.model
    W2VEC_MODEL_FILE = args.embeddingfile
    EMBEDDING_DIM = int(args.dimension)
    SEED = int(args.seed)
    NO_OF_FOLDS = int(args.folds)
    CLASS_WEIGHT = args.class_weight
    N_ESTIMATORS = int(args.estimators)
    LOSS_FUN = args.loss
    KERNEL = args.kernel
    
    print('Word2Vec embedding: %s' %(W2VEC_MODEL_FILE))
    print('Embedding Dimension: %d' %(EMBEDDING_DIM))
    
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_in = path['dir_in']

    word2vec_model = gensim.models.Word2Vec.load(dir_w2v+W2VEC_MODEL_FILE)
    tp = TextProcessor()
    doc_list, tw_class = load_files(dir_in)
    tweets = tp.text_process(doc_list, text_only=True)
    tweets = select_tweets(tweets)

    X, Y = gen_data(tweets, tw_class)

    model = classification_model(X, Y, MODEL_TYPE)
    joblib.dump(model, dir_in + MODEL_TYPE + '.skl')


    # python BoWV.py --model logistic --seed 42 -f model_word2vec -d 100 --folds 10
    # python BoWV.py --model gradient_boosting --seed 42 -f model_word2vec -d 100 --loss deviance --folds 10
    # python BoWV.py --model random_forest --seed 42 -f model_word2vec -d 100 --estimators 20 --folds 10
    # python BoWV.py --model svm_linear --seed 42 -f model_word2vec -d 100 --loss squared_hinge --folds 10
    # python BoWV.py --model svm --seed 42 -f model_word2vec -d 100 --kernel rbf --folds 10
