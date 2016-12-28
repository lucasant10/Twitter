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





def  generate_lda(corpus, dictionary, num_topics):        
    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics,minimum_probability=0.00001, id2word = dictionary,alpha='auto')
    #ldamodel.save('tweet_teste.lda')
    #model = gensim.models.LdaModel.load('android.lda')
    print(ldamodel.print_topics())
    #ldamodel.print_topics()
    return ldamodel


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

dic_words = dict(sort_tfidf_like[1:15000])

list_tw_parl = list()
for parl in parl_tweets:
    temp = list()
    for tw in parl:
        temp.append(list([x for x in tw if x in dic_words]))
    list_tw_parl.append(temp)

text = [list(itertools.chain.from_iterable(t)) for t in list_tw_parl]
corpus, dic = tp.create_corpus(text)
ldamodel = generate_lda(corpus, dic, 5)

for i,parl in enumerate(parl_tweets):
    f =  open(dir_ale+str(i)+"tweets.txt", 'w')
    for tw in parl:
        f.write(" ".join(tw)+"\n")
    f.close()



        



