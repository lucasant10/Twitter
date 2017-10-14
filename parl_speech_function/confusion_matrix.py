import sys
sys.path.append('../')
import configparser
import os
from tfidf import TfIdf
import pickle
import numpy as np
from text_processor import TextProcessor
from collections import Counter
import itertools
import random
import operator
import json


def read_tweets(tw_file):
    tweets = list()
    with open(dir_tw + tw_file) as data_file:
        for line in data_file:
            tweet = json.loads(line)
            tweets.append(tweet['text'])
    return tweets

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = 100*np.asarray(x)
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    dist = np.round(ex / sum_ex, 3) 
    return max(enumerate(dist), key=operator.itemgetter(1))

def nomralization(x):
    x = np.asarray(x)
    dist = np.round(x/x.sum(),3)
    return max(enumerate(dist), key=operator.itemgetter(1))
    
def classifier_s(tweet, w_matrix):
    weight = list()
    for m in w_matrix:
        tmp = 0
        for w in tweet:
            if w in m:
                tmp += m[w]
        weight.append(tmp)
    return softmax(weight)
        
def classifier_n(tweet, w_matrix):
    weight = list()
    for m in w_matrix:
        tmp = 0
        for w in tweet:
            if w in m:
                tmp += m[w]
        weight.append(tmp)
    return nomralization(weight)

def calc_accuracy(conf_matrix):
    t = sum(sum(l) for l in conf_matrix)
    return sum(conf_matrix[i][i] for i in range(len(conf_matrix))) / t
            
if __name__=='__main__':

    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_out = path['dir_out']
    dir_ale = path['dir_ale']
    dir_tw = path['dir_tw']

    print("load tweet files")
    fnames = ([file for root, dirs, files in os.walk(dir_tw)
            for file in files if file.endswith('.json')  ])

    categories_tw = list()
    tp = TextProcessor()
    for fl in fnames:
        categories_tw.append(tp.text_process(read_tweets(fl)))

    categories_counter = list()
    test_data = list()
    for categ in categories_tw:
        k = int(len(categ) * 0.2)
        random.shuffle(categ)
        tmp = list(itertools.chain.from_iterable(categ[k:]))
        categories_counter.append(Counter(tmp))
        test_data.append(categ[:k])


    print("process tfidf")
    tfidf_entropy = list()
    tfidf_smooth = list()
    tfidf_like = list()

    for i , data in enumerate(categories_counter):
        tmp_smooth = dict()
        tmp_like = dict()
        tmp_entropy = dict()
        print("dataset: " + str(i))
        for word in data:
            tf = TfIdf.tf(word, data)
            tmp_entropy[word] = tf * TfIdf.idf_entropy(word, i, categories_counter)
            tmp_smooth[word] = tf * TfIdf.idf_smooth(word, categories_counter)
            tmp_like[word] = tf * TfIdf.idf_like(word, i, categories_counter)
        tfidf_smooth.append(tmp_smooth)
        tfidf_like.append(tmp_like)
        tfidf_entropy.append(tmp_entropy)

    print("processing softmax confusion matrix")
    confusion_like = np.zeros(shape=(len(test_data), len(test_data)))
    confusion_smooth = np.zeros(shape=(len(test_data), len(test_data)))
    confusion_entropy = np.zeros(shape=(len(test_data), len(test_data)))
    for i, data in enumerate(test_data):
        for tw in data:
            j, value = classifier_s(tw, tfidf_like)
            confusion_like[i, j] += 1
            j, value = classifier_s(tw, tfidf_smooth)
            confusion_smooth[i, j] += 1
            j, value = classifier_s(tw, tfidf_entropy)
            confusion_entropy[i, j] += 1
    
    #matrix normalization
    confusion_like = confusion_like / confusion_like.sum(axis=1, keepdims=1)
    confusion_smooth = confusion_smooth / confusion_smooth.sum(axis=1, keepdims=1)
    confusion_entropy = confusion_entropy / confusion_entropy.sum(axis=1, keepdims=1)

    #saving matrix
    header = " ".join([x.split('.')[0] for x in fnames])
    np.savetxt(dir_out+'conf_like_softmax.txt', confusion_like, fmt='%1.2f',delimiter="::",newline="\n", header=header)
    np.savetxt(dir_out+'conf_smooth_softmax.txt', confusion_smooth,fmt='%1.2f',delimiter="::",newline="\n", header=header)
    np.savetxt(dir_out+'conf_entropy_softmax.txt', confusion_entropy, fmt='%1.2f',delimiter="::",newline="\n", header=header)

    print("processing nomralized confusion matrix")
    confusion_like = np.zeros(shape=(len(test_data), len(test_data)))
    confusion_smooth = np.zeros(shape=(len(test_data), len(test_data)))
    confusion_entropy = np.zeros(shape=(len(test_data), len(test_data)))
    for i, data in enumerate(test_data):
        for tw in data:
            j, value = classifier_n(tw, tfidf_like)
            confusion_like[i, j] += 1
            j, value = classifier_n(tw, tfidf_smooth)
            confusion_smooth[i, j] += 1
            j, value = classifier_n(tw, tfidf_entropy)
            confusion_entropy[i, j] += 1

    #matrix normalization            
    confusion_like = confusion_like / confusion_like.sum(axis=1, keepdims=1)
    confusion_smooth = confusion_smooth / confusion_smooth.sum(axis=1, keepdims=1)
    confusion_entropy = confusion_entropy / confusion_entropy.sum(axis=1, keepdims=1)

    #saving matrix
    np.savetxt(dir_out+'conf_like_normal.txt', confusion_like, fmt='%1.2f',delimiter="::",newline="\n", header=header)
    np.savetxt(dir_out+'conf_smooth_normal.txt', confusion_smooth, fmt='%1.2f',delimiter="::",newline="\n", header=header)
    np.savetxt(dir_out+'conf_entropy_normal.txt', confusion_entropy, fmt='%1.2f',delimiter="::",newline="\n", header=header)
