import math
from collections import Counter
from PtBrTwitter import PtBrTwitter 
import json
import os
from text_processor import TextProcessor
import itertools
import pickle
import math
from tfidf import TfIdf
from scipy import stats
from itertools import islice
import matplotlib.pyplot as plt


def loadCounters(dir_in):
    counter_list = list()
    tot_counter = Counter()
    pck = ([file for root, dirs, files in os.walk(dir_in)
            for file in files if file.endswith('.pck') ])
    for i,counter_file in enumerate(pck):
        print("processando o arquivo: "+counter_file+"\n")
        with open(dir_in+counter_file, 'rb') as data_file:
            tw_counter = pickle.load(data_file)
            tot_counter += tw_counter
            counter_list.append(tw_counter)
    return tot_counter,counter_list



if __name__=='__main__':

    dir_in = "/home/lucasso/Documents/random_pck/"
    dir_out = "/home/lucasso/Documents/"
    file_parl = "/home/lucasso/Documents/random_pck/deputados.pck"
    tfidf_n = list()
    tf_log_idf = list()
    tfidf_like = list() 
    corr = ""

    with open(file_parl, 'rb') as handle:
        parl_counter = pickle.load(handle)

    tot_counter,counter_list = loadCounters(dir_in)
    tfidf = TfIdf()
    for word in parl_counter:
        tf = tfidf.tf(word, parl_counter)
        idf = tfidf.idf(word,counter_list)
        log_idf = tfidf.idf_smooth(word,counter_list)
        ent_idf = tfidf.idf_like(word,parl_counter, tot_counter, counter_list)
        
        tfidf_n.append(tf*idf)
        tf_log_idf.append(tf*log_idf)
        tfidf_like.append(tf*ent_idf)
"""
    corr +=  "tfidf X tfidf_smooth: "+str(stats.spearmanr(tfidf_n,tf_log_idf))+"\n"
    corr +=  "tfidf X tfidf_like: "+str(stats.spearmanr(tfidf_n,tfidf_like))+"\n"
    corr +=  "tfidf_like X tfidf_smooth: "+str(stats.spearmanr(tfidf_like,tf_log_idf))+"\n"

    corr +=  "tfidf X tfidf_smooth: "+str(stats.pearsonr(tfidf_n,tf_log_idf))+"\n"
    corr +=  "tfidf X tfidf_like: "+str(stats.pearsonr(tfidf_n,tfidf_like))+"\n"
    corr +=  "tfidf_like X tfidf_smooth: "+str(stats.pearsonr(tfidf_like,tf_log_idf))+"\n"

    with open(dir_out+"saida_correlacao2.txt", "w") as f:
        f.write(corr)
        f.close()

"""
    dic_tfidf= dict(zip(parl_counter.keys(), tfidf_n))
    dic_tf_log_idf= dict(zip(parl_counter.keys(), tf_log_idf))
    dic_tfidf_like= dict(zip(parl_counter.keys(), tfidf_like))

    dic_tfidf = dict(sorted(dic_tfidf.items(), key=lambda x: x[1], reverse=True))
    dic_tf_log_idf = dict(sorted(dic_tf_log_idf.items(), key=lambda x: x[1], reverse=True))
    dic_tfidf_like = dict(sorted(dic_tfidf_like.items(), key=lambda x: x[1], reverse=True))

    plot = list()
    for i in range(25,1000,25):
        #retorna os n primeiros elementos do dicionario. entao é computado a correlacao dos valores
        s1 = stats.spearmanr(list(dict(islice(dic_tfidf.items(),i)).values()) ,list(dict(islice(dic_tf_log_idf.items(),i)).values()))[0]
        s2 = stats.spearmanr(list(dict(islice(dic_tfidf.items(),i))),list(dict(islice(dic_tfidf_like.items(),i)).values()))[0]
        s3 = stats.spearmanr(list(dict(islice(dic_tfidf_like.items(),i)).values()) , list(dict(islice(dic_tf_log_idf.items(),i)).values()))[0]
        plot.append((s1,s2,s3))

    for temp in zip(*plot):
        plt.plot(range(25,1000,25), temp)

    plt.title('Correlação de Spearman para as K top tfidf')
    plt.xlabel('k top tfidf')
    plt.ylabel('Spearman')
    plt.legend(['tfidf X tfidf_smooth', 'tfidf x tfidf_like', 'tfidf_like x tfidf_smooth'])


    


       