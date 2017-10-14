import sys
sys.path.append('../')
from collections import Counter
import configparser
import os
from tfidf import TfIdf
import pickle
import itertools
import random

def remove_irrelevant(counter):
    tmp = dict()
    for i, v in counter.items():
        if v > 1:
            tmp[i] = v
    return Counter(tmp)


if __name__=='__main__':

    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_out = path['dir_out']
    dir_ale = path['dir_ale']

    #parliamentary tweets
    with open(dir_out+"list_parl_tw_processed.pck", 'rb') as data_file:
        tweets = pickle.load(data_file)

    print("create a tweet counter")
    tweets = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(tweets))))
    
    dataset = [remove_irrelevant(Counter(tweets))]

    #load dataset tweets counter
    with open(dir_ale+"coleta1.pck", 'rb') as data_file:
        dataset.append(remove_irrelevant(pickle.load(data_file)))

    with open(dir_ale+"coleta2.pck", 'rb') as data_file:
        dataset.append(remove_irrelevant(pickle.load(data_file)))

    with open(dir_ale+"coleta3.pck", 'rb') as data_file:
        dataset.append(remove_irrelevant(pickle.load(data_file)))


    print("process tfidf")
    tfidf_entropy = list()
    tfidf_smooth = list()
    tfidf_like = list()

    for i , data in enumerate(dataset):
        tmp_smooth = dict()
        tmp_like = dict()
        tmp_entropy = dict()
        print("dataset: " + str(i))
        for word in data:
            tf = TfIdf.tf(word, data)
            tmp_entropy[word] = tf * TfIdf.idf_entropy(word, i, dataset)
            tmp_smooth[word] = tf * TfIdf.idf_smooth(word, dataset)
            tmp_like[word] = tf * TfIdf.idf_like(word, i, dataset)
        tfidf_smooth.append(tmp_smooth)
        tfidf_like.append(tmp_like)
        tfidf_entropy.append(tmp_entropy)


    print("save tfidf")
    with open(dir_out+"tfidf_entropy.pck", 'wb') as handle:
        pickle.dump(tfidf_entropy, handle)

    with open(dir_out+"tfidf_smooth.pck", 'wb') as handle:
        pickle.dump(tfidf_smooth, handle)
    
    with open(dir_out+"tfidf_like.pck", 'wb') as handle:
        pickle.dump(tfidf_like, handle) 



 






