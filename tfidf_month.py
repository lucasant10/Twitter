import os
from text_processor import TextProcessor
import json
import itertools
import pickle
import configparser
import re
import math
import codecs
from collections import Counter
from tfidf import TfIdf

def month_tw(time):
    first = 1224550880000
    month= 2419200000
    return math.ceil((time - first)/month)

def save_file(path,tweet, parl , month):
    filename = path+"tw_month/month_"+str(month)+"/dep_"+str(parl)+".txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = codecs.open(filename, 'a', 'utf-8')
    f.write(tweet+os.linesep)
    f.close()

def docs_counters(random_list,tot_counter):
    docs_counter =list()
    random_dic = dict()
    docs_counter.append(tot_counter)
    for r in random_pck:
        random_dic = dict(random_dic.items() | r.items())
        x = math.floor(len(random_dic)/len(tot_counter))
    n=0
    tmp = list()
    for i,k in enumerate(random_dic.items()):
        if(i//x != n):
            docs_counter.append(dict(tmp))
            tmp=list()
            tmp.append(k)
            n=i//x
        tmp.append(k)
    docs_counter.append(dict(tmp))
    return docs_counter


def load_files(path):
    txt = ([file for root, dirs, files in os.walk(path)
        for file in files if file.endswith('.txt') ])
    doc_list = list()
    for m in txt:
        dp = list()
        with open(path+m,"rb") as data_file:
            for line in data_file:
                dp.append(line.decode('utf-8'))
        doc_list.append(dp)
    return doc_list

def tfidf_month(tw_month,random_list):
    tweets = list(itertools.chain.from_iterable(itertools.chain.from_iterable(tw_month)))
    tot_counter = Counter(tweets)
    dep_counts = list()
    for dep in tw_month:
        tw = list(itertools.chain.from_iterable(dep))
        print(tw)
        dep_counts.append(Counter(tw))
    docs_counter = docs_counters(random_list,tot_counter)
    tfidf = TfIdf()
    tfidf_like = list()
    for word in tot_counter:
        tfidf_like.append(tfidf.tf(word,tot_counter)*tfidf.idf_like(word,tot_counter,tot_counter,docs_counter, dep_counts))
    sort_tfidf_like = list(zip(tot_counter.keys(), tfidf_like))
    sort_tfidf_like = sorted(sort_tfidf_like, key=lambda x: x[1], reverse=True)
    return sort_tfidf_like

def save_pck(path, tw_file):
    with open(path+"ranked_tfidf.pck", 'wb') as handle:
        pickle.dump(tw_file, handle)
    filename = path+"ranked_tfidf.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = codecs.open(filename, 'w', 'utf-8')
    for w,i in tw_file[:100]:
        f.write(w+os.linesep)
    f.close()



if __name__=='__main__':

cf = configparser.ConfigParser()
cf.read("file_path.properties")
path = dict(cf.items("file_path"))
dir_in = path['dir_in']
dir_out = path['dir_out']


with open(dir_out+"time_doc_parl_processed.pck",'rb') as handle:
    parl_tw = pickle.load(handle)

first = 1224550880000
last = 1454803200000
month= 2419200000
week = 604800000 


tw_dic = dict()
for i,dep in enumerate(parl_tw):
    for tw in dep:
        tw_dic[str(month_tw(tw[0]))+"_"+str(i)]=tw[1]
        save_file(dir_out,tw[1],i,month_tw(tw[0]))

random_pck =list()
with open(dir_out+"random-pck/coleta1.pck",'rb') as handle:
    random_pck.append(pickle.load(handle))

with open(dir_out+"random-pck/coleta2.pck",'rb') as handle:
    random_pck.append(pickle.load(handle))

month_files = list()
for m in range(10):
    month_files.append(load_files(dir_out+"tw_month/month_"+str(m)+"/"))

tp = TextProcessor()
month_processed =list()
for tw in month_files:
    tmp = list()
    for dep in tw:
        tmp.append(tp.text_process(dep,text_only=True))
    month_processed.append(tmp)

ranked_month = list()
for i,month in enumerate(month_processed):
    tmp = tfidf_month(month,random_pck)
    ranked_month.append(tmp)
    save_pck(dir_out+"tw_month/month_"+str(i)+"/",tmp)

tfidf = TfIdf()


#calcular o tfidf para cada pasta com aleatorio dependendo do tamanho de cada mes 



        



