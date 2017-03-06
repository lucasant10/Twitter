import nltk
from nltk.collocations import *
import os
import pickle
from text_processor import TextProcessor
import json
from tfidf import TfIdf
import configparser

def add_separator(tweets):
    separator_word = "STOP"
    tweet_words = [separator_word]
    for tweet_tokens in tweets:
        tweet_words += tweet_tokens
        tweet_words.append(separator_word)
    return tweet_words

def load_words_entropy(dir_in, file_name):
    with open(dir_in+file_name, 'rb') as handle:
            word_list = pickle.load(handle)
            return word_list


def get_bigrams(tokens, n, fq_filter=False):
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigram_finder.apply_ngram_filter(is_separator_bigram)
    if fq_filter:
        bigram_finder.apply_freq_filter(n)   
    return bigram_finder

def get_trigrams(tokens, n, fq_filter=False):
    finder = TrigramCollocationFinder.from_words(tokens)
    finder.apply_ngram_filter(is_separator_trigram)
    if fq_filter:
        finder.apply_freq_filter(n)   
    return finder


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

def save_pck(dir_out, file_name, word_list):
    with open(dir_out+file_name, 'wb') as handle:
        pickle.dump(word_list, handle)

def is_separator_bigram(w1,w2):
    separator_word = "STOP"
    return  w1  == separator_word or w2 == separator_word

def is_separator_trigram(w1,w2,w3):
    separator_word = "STOP"
    return  w1  == separator_word or w2 == separator_word or w3 == separator_word

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
    tweets = tp.text_process(doc_list,text_only=True)
    tw_words = add_separator(tweets)
    parl_bigrams = get_bigrams(tw_words,3,True)

    #processa os tweets de cada deputado
    parl_processed = list()
    parl_tri_processed = list()
    for l in parl_tw_list:
        temp = add_separator(tp.text_process(l,text_only=True))
        parl_tri_processed.append(get_trigrams(temp, 3, True))
        parl_processed.append(get_bigrams(temp,3,True))

    with open(dir_out+"list_dept_bigrams_.pck", 'wb') as handle:
        pickle.dump(parl_processed, handle)
    with open(dir_out+"list_dept_trigrams_.pck", 'wb') as handle:
        pickle.dump(parl_tri_processed, handle)

    #processa os bigramas dos tweets dos documentos aleatorios
    alea_processed = list()
    alea_tri_processed = list()
    for l in list_aleatory:
        temp = add_separator(tp.text_process(l,text_only=True))
        alea_tri_processed.append(get_trigrams(temp, 3,True))
        alea_processed.append(get_bigrams(temp,3,True))

    with open(dir_out+"list_alea_bigrams.pck", 'wb') as handle:
        pickle.dump(alea_processed, handle)
    with open(dir_out+"list_alea_trigrams.pck", 'wb') as handle:
        pickle.dump(alea_tri_processed, handle)

    

    bgr_counter = parl_bigrams.ngram_fd
    parl_bgr_counter = [l.ngram_fd for l in parl_processed]
    docs_bgr_counter = [l.ngram_fd for l in alea_processed]
    docs_bgr_counter.append(bgr_counter)


    tfidf = TfIdf()
    tfidf_smooth = list() 
    for bgr in bgr_counter:
        tfidf_smooth.append(tfidf.tf(bgr,bgr_counter)*tfidf.idf_smooth(bgr,docs_bgr_counter))

    dic_tfidf_smooth = list(zip(bgr_counter.keys(), tfidf_smooth))
    dic_tfidf_smooth = sorted(dic_tfidf_smooth, key=lambda x: x[1], reverse=True)

    tot_counter = dict()
    for y in docs_bgr_counter:
        for k in y.keys(): tot_counter[k] = k in tot_counter and tot_counter[k]+y[k] or y[k]

tfidf_like = list()
for bgr in bgr_counter:
    tfidf_like.append(tfidf.tf(bgr,bgr_counter)*tfidf.idf_like(bgr,bgr_counter,tot_counter,docs_bgr_counter, parl_bgr_counter))

dic_tfidf_like = list(zip(bgr_counter.keys(), tfidf_like))
dic_tfidf_like = sorted(dic_tfidf_like, key=lambda x: x[1], reverse=True)


    #processa os trigramas dos tweets dos documentos aleatorios
    alea_tri_processed = list()
    for l in list_aleatory:
        temp = add_separator(tp.text_process(l,text_only=True))
        temp = get_trigrams(temp,2,True)
        alea_tri_processed.append(temp)
    with open(dir_out+"list_alea_trigrams.pck", 'wb') as handle:
        pickle.dump(alea_tri_processed, handle)


    _ , parl_tw_list = load_files(dir_in)
    #processa os trigramas dos tweets de cada deputado
    parl_tri_processed = list()
    for l in parl_tw_list:
        temp = add_separator(tp.text_process(l,text_only=True))
        temp = get_trigrams(temp,3,True)
        parl_tri_processed.append(temp)
    with open(dir_out+"list_parl_trigrams.pck", 'wb') as handle:
        pickle.dump(parl_tri_processed, handle)


    tri_counter = parl_trigrams.ngram_fd
    parl_tri_counter = [l.ngram_fd for l in parl_tri_processed]
    docs_tri_counter = [l.ngram_fd for l in alea_tri_processed]
    docs_tri_counter.append(tri_counter)


    tot_counter = dict()
    for y in docs_tri_counter:
        for k in y.keys(): tot_counter[k] = k in tot_counter and tot_counter[k]+y[k] or y[k]


    tfidf_like_tri = list()
    for tri in tri_counter:
        tfidf_like_tri.append(tfidf.tf(tri,tri_counter)*tfidf.idf_like(tri,tri_counter,tot_counter,docs_tri_counter, parl_tri_counter))

    tri_tfidf_like = list(zip(tri_counter.keys(), tfidf_like_tri))
    tri_tfidf_like = sorted(tri_tfidf_like, key=lambda x: x[1], reverse=True)


#processa os tweets de cada deputado
parl_processed = list()
for parl in parl_tw_processed:
    temp = add_separator(parl)
    temp = get_bigrams(temp,True)
    parl_processed.append(temp)

with open(dir_rob+"bigram_tfidf_like.pck", 'wb') as handle:
    pickle.dump(dic_tfidf_like, handle)


parl_bigrams = list(itertools.chain.from_iterable(parl_tw_processed))

bgr_counter = dict()
for i in parl_processed:
   bgr_counter.update(i.ngram_fd)

with open(dir_out+"list_alea_bigrams.pck",'rb') as handle:
    alea_processed = pickle.load(handle)


f =  open(dir_rob+"10k_bigram_tfidf_like.txt", 'w')
for w,i in dic_tfidf_like[:10000]:
    f.write("_".join(w)+","+str(i)+" \n")

f.close()


bgr_counter = {}
for y in parl_bgr_counter:
    for k in y.keys(): bgr_counter[k] = k in bgr_counter and bgr_counter[k]+y[k] or y[k]
