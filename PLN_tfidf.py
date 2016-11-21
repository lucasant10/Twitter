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
    tfidf_n = tf_log_idf = tfidf_like = list() 
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
corr +=  "tfidf X tfidf_smooth: "+str(stats.spearmanr(tfidf_n,tf_log_idf))+"\n"
corr +=  "tfidf X tfidf_like: "+str(stats.spearmanr(tfidf_n,tfidf_like))+"\n"
corr +=  "tfidf_like X tfidf_smooth: "+str(stats.spearmanr(tfidf_like,tf_log_idf))+"\n"

corr +=  "tfidf X tfidf_smooth: "+str(stats.pearsonr(tfidf_n,tf_log_idf))+"\n"
corr +=  "tfidf X tfidf_like: "+str(stats.pearsonr(tfidf_n,tfidf_like))+"\n"
corr +=  "tfidf_like X tfidf_smooth: "+str(stats.pearsonr(tfidf_like,tf_log_idf))+"\n"

with open(dir_out+"saida_correlacao2.txt", "w") as f:
    f.write(corr)
    f.close()


with open(dir_out+"tfidf_n.pck", 'wb') as handle:
    pickle.dump(tfidf_n, handle)

with open(dir_out+"tf_log_idf.pck", 'wb') as handle:
    pickle.dump(tf_log_idf, handle)
    
with open(dir_out+"tfidf_like.pck", 'wb') as handle:
    pickle.dump(tfidf_like, handle)
       