from collections import Counter
import json
import os
from text_processor import TextProcessor
import itertools
import pickle
import configparser
from tfidf import TfIdf



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


if __name__=='__main__':

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']
    dir_out = path['dir_out']
    dir_ale = path['dir_ale']
    dir_rob = path['dir_rob']

    doc_list, parl_tw_list = load_files(dir_rob)
    tp = TextProcessor()

    parl_tw_processed = list()
    for l in parl_tw_list:
        parl_tw_processed.append(tp.text_process(l, text_only=True))


    with open(dir_in+"coleta1.pck",'rb') as handle:
        coleta1 = pickle.load(handle)

    with open(dir_in+"coleta2.pck",'rb') as handle:
        coleta2 = pickle.load(handle)

    tweets = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(parl_tw_processed))))
    tot_counter = Counter(tweets)


    parl_counters = list()
    for parl in parl_tw_processed:
        tw = list(itertools.chain.from_iterable(parl))
        parl_counters.append(Counter(tw))


    docs_counter =list()
    docs_counter.append(tot_counter)
    docs_counter.append(coleta1)
    docs_counter.append(coleta2)

    tfidf = TfIdf()

    tfidf_like = list()
    for word in tot_counter:
        tfidf_like.append(tfidf.tf(word,tot_counter)*tfidf.idf_like(word,tot_counter,tot_counter,docs_counter, parl_counters))

    sort_tfidf_like = list(zip(tot_counter.keys(), tfidf_like))
    sort_tfidf_like = sorted(sort_tfidf_like, key=lambda x: x[1], reverse=True)

    with open(dir_rob+"sort_tfidf_like.pck", 'wb') as handle:
        pickle.dump(sort_tfidf_like, handle)

    with open(dir_rob+"tfidf_like.pck", 'wb') as handle:
        pickle.dump(tfidf_like, handle)

    with open(dir_rob+"parl_tw_processed.pck", 'wb') as handle:
        pickle.dump(parl_tw_processed, handle)

    f =  open(dir_rob+"10k_tfidf_like.txt", 'w')
    for w,i in sort_tfidf_like[:10000]:
        f.write(w+", \n")

    f.close()

