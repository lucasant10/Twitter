import nltk
from nltk.collocations import *
import os
import numpy as np
from text_processor import TextProcessor
import json
import itertools
import pickle
from tfidf import TfIdf
import configparser
from collections import Counter
import re


def replace_ngram(ngram,sentence):
    word =" ".join(ngram)
    word_r = "_".join(ngram)
    s = re.sub(r'\b%s\b' % word,word_r,sentence)
    return s

def load_files(dir_in):
    doc_list=list()
    tw_files = ([file for root, dirs, files in os.walk(dir_in)
        for file in files if file.endswith('.json') ])
    parl_tw_list = list()
    for tw_file in tw_files:
        temp=list()
        with open(dir_in+tw_file) as data_file:
            for line in data_file:
                tweet = json.loads(line)
                temp.append(tweet['text'])
                doc_list.append(tweet['text'])           
        parl_tw_list.append(temp)
    return doc_list, parl_tw_list 

def trigram_word_check(p1,p2,p3, tw_list):
    tri = "_".join([p1,p2,p3])
    new_list = list()
    l = len(tw_list)
    if l >= 3:
        i=0
        while(i < l):
            print(i)
            if(tw_list[i]==p1 and tw_list[i+1]==p2 and tw_list[i+2]==p3):
                new_list.append(tri)
                i+=3                
            elif(i>=l-3):
                new_list = new_list + tw_list[i:] #last 3 elements
                i+=len(tw_list[i:])
            else:
                new_list.append(tw_list[i])  
                i+=1              
    else:
        return tw_list
    return new_list



if __name__=='__main__':

cf = configparser.ConfigParser()
cf.read("file_path.properties")
path = dict(cf.items("file_path"))
dir_in = path['dir_in']
dir_out = path['dir_out']
dir_ale = path['dir_ale']
dir_pck = path['dir_pck']


doc_list, parl_tw_list = load_files(dir_in)
_ ,list_aleatory = load_files(dir_ale)

tp = TextProcessor()
tweets = tp.text_process(doc_list, text_only=True)

parl_tw_processed = list()
for l in parl_tw_list:
    parl_tw_processed.append(tp.text_process(l, text_only=True))

alea_tw_processed = list()
for l in list_aleatory:
    alea_tw_processed.append(tp.text_process(l, text_only=True))

for i,l in enumerate(alea_tw_processed):
    alea_tw_processed[i] = [n for n in l if n]

with open(dir_out+"bgr_tfidf_like.pck",'rb') as handle:
    parl_bigrams = pickle.load(handle)

with open(dir_out+"tri_tfidf_like.pck",'rb') as handle:
    parl_trigrams = pickle.load(handle)

tri_list = dict(parl_trigrams[:3000]).keys()
bgr_list = dict(parl_bigrams[:10000]).keys()

parl_tweets = list()
for parl in parl_tw_processed:
    for i,tw in enumerate(parl):
        print("palavra # :"+str(i))
        s = " ".join(tw)
        for trigram in tri_list:    
            s = replace_ngram(trigram, s)
        print("saiu trigrama")
        for bigram in bgr_list:
            s = replace_ngram(bigram,s)
        print("saiu bigrama")
        parl[i] = s.split(" ")
    parl_tweets.append(parl)


alea_tweets = list()
for alea in alea_tw_processed:
    for i,tw in enumerate(alea):
        print("palavra # :"+str(i))
        s = " ".join(tw)
        for trigram in tri_list:    
            s = replace_ngram(trigram, s)
        print("saiu trigrama")
        for bigram in bgr_list:
            s = replace_ngram(bigram,s)
        print("saiu bigrama")
        alea[i] = s.split(" ")
    alea_tweets.append(alea)


tweets = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(parl_tweets))))
tot_counter = Counter(tweets)

docs_counter = list()
for alea_tw in alea_tweets:
    tw = list(itertools.chain.from_iterable(alea_tw))
    docs_counter.append(Counter(tw))
docs_counter.append(tot_counter)

parl_counters = list()
for parl in parl_tweets:
    tw = list(itertools.chain.from_iterable(parl))
    parl_counters.append(Counter(tw))

tfidf = TfIdf()
tfidf_like_bi_trigrams = list()
for word in tot_counter:
    tfidf_like_bi_trigrams.append(tfidf.tf(word,tot_counter)*tfidf.idf_like(word,tot_counter,tot_counter,docs_counter, parl_counters))

sort_tfidf_like = list(zip(tot_counter.keys(), tfidf_like_bi_trigrams))
sort_tfidf_like = sorted(sort_tfidf_like, key=lambda x: x[1], reverse=True)

with open(dir_out+"sort_tfidf_like_bi_trigram.pck", 'wb') as handle:
    pickle.dump(sort_tfidf_like, handle)




