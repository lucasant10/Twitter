import nltk
from nltk.collocations import *
import os
import pickle
import numpy as np
from text_processor import TextProcessor
import json
import itertools
import pickle
from tfidf import TfIdf
import configparser
from collections import Counter






if __name__=='__main__':

cf = configparser.ConfigParser()
cf.read("file_path.properties")
path = dict(cf.items("file_path"))
dir_in = path['dir_in']
dir_out = path['dir_out']
dir_ale = path['dir_ale']

tp = TextProcessor()

with open(dir_out+"list_parl_tw_bi_trigrams2.pck",'rb') as handle:
    parl_tweets = pickle.load(handle)

with open(dir_out+"tfidf_like_bi_trigrams.pck",'rb') as handle:
    tfidf_like_bi_trigrams = pickle.load(handle)

dic_words = dict(sort_tfidf_like[:20000])

list_tw_parl = list()
for parl in parl_tweets:
    temp = list()
    for tw in parl:
        temp.append(list([x for x in tw if x in dic_words]))
    list_tw_parl.append(temp)



