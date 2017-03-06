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
from scipy.stats import beta,expon
import csv
import math

def idf_beta( word,parl_counter, tot_counter,doc_counter, counter_list_parl,b1,b2):
        h_max = math.log2(len(doc_counter))
        h_word = TfIdf.entropy(word,tot_counter,doc_counter)
        x = math.pow(2,h_word)/math.pow(2,h_max)
        return ((h_max-h_word)
            *TfIdf.parl_prob(word,parl_counter,doc_counter)*beta.pdf(x,b1,b2))

def idf_pow( word,parl_counter, tot_counter,doc_counter, counter_list_parl,b1,b2):
        h_max = math.log2(len(doc_counter))
        h_word = TfIdf.entropy(word,tot_counter,doc_counter)
        x = math.pow(2,h_word)/math.pow(2,h_max)
        return (expon.pdf(h_word,scale=0.2)
            *TfIdf.parl_prob(word,parl_counter,doc_counter)*beta.pdf(x,b1,b2))

def idf_like( word,parl_counter, tot_counter,doc_counter, counter_list_parl):
    return ((math.log2(len(doc_counter))-TfIdf.entropy(word,tot_counter,doc_counter))
        *TfIdf.parl_prob(word,parl_counter,doc_counter)*TfIdf.parl_entropy(word, tot_counter, counter_list_parl))


if __name__=='__main__':

cf = configparser.ConfigParser()
cf.read("file_path.properties")
path = dict(cf.items("file_path"))
dir_in = path['dir_in']
dir_out = path['dir_out']
dir_ale = path['dir_ale']
dir_pck = path['dir_pck']
tfidf = TfIdf()

with open(dir_out+"list_alea_bigrams.pck",'rb') as handle:
    ale_tweets = pickle.load(handle)
with open(dir_out+"list_dept_bigrams_.pck",'rb') as handle:
    parl_tweets = pickle.load(handle)

parl_bgr_counter = [l.ngram_fd for l in parl_tweets]
docs_bgr_counter = [l.ngram_fd for l in ale_tweets]
bgr_counter = dict()
for y in parl_bgr_counter:
    for k in y.keys(): bgr_counter[k] = k in bgr_counter and bgr_counter[k]+y[k] or y[k]

docs_bgr_counter.append(bgr_counter)

tot_counter = dict()
for y in docs_bgr_counter:
    for k in y.keys(): tot_counter[k] = k in tot_counter and tot_counter[k]+y[k] or y[k]

f = open(dir_out+'param_beta.csv', 'rt')
reader = csv.DictReader(f)
params=list()
for row in reader:
    like = list()
    beta = list()
    exp = list()
    for bgr in bgr_counter:
        freq = tfidf.tf(bgr,bgr_counter)
        like.append(freq
            *idf_like(bgr,bgr_counter,tot_counter,docs_bgr_counter, parl_bgr_counter))
        beta.append(freq
            *idf_beta(bgr,bgr_counter,tot_counter,docs_bgr_counter, parl_bgr_counter,float(row["b1"]),float(row["b2"])))
        exp.append(freq
            *idf_pow(bgr,bgr_counter,tot_counter,docs_bgr_counter, parl_bgr_counter,float(row["b1"]),float(row["b2"])))
    like_list = list(zip(bgr_counter.keys(), like))
    like_list = sorted(like_list, key=lambda x: x[1], reverse=True)
    beta_list = list(zip(bgr_counter.keys(), beta))
    beta_list = sorted(beta_list, key=lambda x: x[1], reverse=True)
    pow_list = list(zip(bgr_counter.keys(), exp))
    pow_list = sorted(pow_list, key=lambda x: x[1], reverse=True)
    params.append(row["param"])
    with open(dir_pck+str(row["param"])+"_like"+".pck", 'wb') as handle:
        pickle.dump(like_list, handle)
    with open(dir_pck+str(row["param"])+"_beta"+".pck", 'wb') as handle:
        pickle.dump(beta_list, handle)
    with open(dir_pck+str(row["param"])+"_pow"+".pck", 'wb') as handle:
        pickle.dump(pow_list, handle)

f.close()

csvfile = open(dir_out+'compare_table.csv', 'w')

pck = ([file for root, dirs, files in os.walk(dir_pck)
        for file in files if file.endswith('.pck') ])
ranked_list = list()
for p in pck:
    with open(dir_pck+p,'rb') as handle:
        ranked_list.append(pickle.load(handle))
ranked = [["_".join(i) for i,v in x[:1000]] for x in ranked_list]
csvfile.write(", ".join(pck)+ "\n")
for values in zip(*ranked):
    csvfile.write(", ".join(values)+"\n")

csvfile.close()















