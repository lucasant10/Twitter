import sys
sys.path.append('../')
import configparser
import os
from tfidf import TfIdf
import pickle
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = 1000 * np.asarray(x)
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return np.round(ex / sum_ex, 3) 

def nomralization(x):
    x = np.asarray(x)
    return np.round(x/x.sum(),3)

def classifier_s(tweet, w_matrix):
    tweet = tweet.split(" ")
    weight = list()
    for m in w_matrix:
        tmp = 0
        for w in tweet:
            if w in m:
                tmp += m[w]
        weight.append(tmp)
    return softmax(weight)
        
def classifier_n(tweet, w_matrix):
    tweet = tweet.split(" ")
    weight = list()
    for m in w_matrix:
        tmp = 0
        for w in tweet:
            if w in m:
                tmp += m[w]
        weight.append(tmp)
    return nomralization(weight)


if __name__=='__main__':

    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_out = path['dir_out']
    dir_ale = path['dir_ale']

    #load tfidf_smooth from datasets
    with open(dir_out+"tfidf_smooth.pck", 'rb') as data_file:
        tfidf_smooth = pickle.load(data_file)

    #load tfidf_like from datasets
    with open(dir_out+"tfidf_like.pck", 'rb') as data_file:
        tfidf_like = pickle.load(data_file)

    #load tfidf_like from datasets
    with open(dir_out+"tfidf_entropy.pck", 'rb') as data_file:
        tfidf_entropy = pickle.load(data_file)

    txt1 = ""
    txt2 = ""
    txt3 = ""
    tw = open(dir_out + "tweets.txt", "r")
    for l in tw:
         txt1 += l+"softmax"+str(classifier_s(l,tfidf_like)) +"\nnomralization"+str(classifier_n(l,tfidf_like))+"\n\n"
         txt2 += l+"softmax"+str(classifier_s(l,tfidf_smooth)) +"\nnomralization"+str(classifier_n(l,tfidf_smooth))+"\n\n"
         txt3 += l+"softmax"+str(classifier_s(l,tfidf_entropy)) +"\nnomralization"+str(classifier_n(l,tfidf_entropy))+"\n\n"

    f =  open(dir_out+"tfidf_like_classifier.txt", 'w')
    f.write(txt1)
    f.close()

    f =  open(dir_out+"tfidf_smooth_classifier.txt", 'w')
    f.write(txt2)
    f.close()

    f =  open(dir_out+"tfidf_entropy_classifier.txt", 'w')
    f.write(txt3)
    f.close()


#TODO: consertar o tfidf lda e treinar com o dataset do mesmo tamanho.