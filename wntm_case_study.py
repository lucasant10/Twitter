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
from gensim import corpora, matutils
import gensim



if __name__=='__main__':

cf = configparser.ConfigParser()
cf.read("file_path.properties")
path = dict(cf.items("file_path"))
dir_in = path['dir_in']
dir_out = path['dir_out']
dir_ale = path['dir_ale']
tp = TextProcessor()

texts = list()
adj_mtx = open(dir_out+"exemplo.adjacent", "r") 
for l in adj_mtx:
    texts.append(l)
corpus, dic = tp.create_corpus(texts)