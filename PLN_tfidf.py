# -*- coding: utf-8 -*-
from collections import Counter
import json
import os
from text_processor import TextProcessor
import pickle
from tfidf import TfIdf
from scipy import stats
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from text_processor import TextProcessor
import itertools

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

def plot_cloud(lista, n, name):
    sliced = [(i,v) for i,v in lista[0:n]]
    txt=""
    for i,k in sliced:
        if(i in parl_counter):
            for a in range(int(k*10000)):
                txt += " "+i
    wc = WordCloud().generate(txt)
    plt.imshow(wc)
    plt.savefig(dir_out+name+'.png', dpi=300)
    plt.clf()

def plot_dep_cloud(tw_list,lista, n, name):
    sliced = [(i,v) for i,v in lista[0:n] if i in tw_list]
    txt=""
    for i,k in sliced:
        if(i in parl_counter):
            for a in range(int(k*100000)):
                txt += " "+i
    wc = WordCloud().generate(txt)
    plt.imshow(wc)
    plt.savefig(dir_out+name+'.png', dpi=300)
    plt.clf()

def list_intersect(x, y):
    return list(set(x).intersection(y))


def plot_tfidfs(sort_tfidf, sort_tf_log_idf,sort_tfidf_like):
    plot = list()
    for i in range(25,10000,25):
        l1 = sort_tfidf[0:i]
        l2 = sort_tf_log_idf[0:i]
        l3 = sort_tfidf_like[0:i]
        #retorna a intersecao da lista

        inter1 = list_intersect([i for i,v in l1],[i for i,v in l2])
        inter2 = list_intersect([i for i,v in l1],[i for i,v in l3])
        inter3 = list_intersect([i for i,v in l2],[i for i,v in l3])
        #pega os elementos da lista, ordenados pela interseção, então é computado a correlacao dos valores

        tmp1 = list(itertools.chain.from_iterable([[v for k,v in l1 if i==k ] for i in inter1 ]))
        tmp2 = list(itertools.chain.from_iterable([[v for k,v in l2 if i==k ] for i in inter1 ]))
        s1 = stats.spearmanr(tmp1,tmp2)[0]
        
        tmp1 = list(itertools.chain.from_iterable([[v for k,v in l1 if i==k ] for i in inter2 ]))
        tmp2 = list(itertools.chain.from_iterable([[v for k,v in l3 if i==k ] for i in inter2 ]))
        s2 = stats.spearmanr(tmp1,tmp2)[0]
        
        tmp1 = list(itertools.chain.from_iterable([[v for k,v in l3 if i==k ] for i in inter3 ]))
        tmp2 = list(itertools.chain.from_iterable([[v for k,v in l2 if i==k ] for i in inter3 ]))
        s3 = stats.spearmanr(tmp1,tmp2)[0]
        plot.append((s1,s2,s3))

    tf, log, like = zip(*plot)
    tfidf_colors_label = (
        (tf,'green','tfidf X tfidf_smooth'),
        (log,'red','tfidf x tfidf_like'),
        (like,'blue','tfidf_like x tfidf_smooth')
        )
    for v_tfidf, color, label in tfidf_colors_label:
        plt.plot(range(1,10000,25), v_tfidf, color=color, label=label)  
    plt.title('Correlacao de Spearman para as K top tfidf')
    plt.xlabel('k top tfidf')
    plt.ylabel('Spearman')
    plt.legend()
    plt.savefig(dir_out+'tfidf.png', dpi=300)
    plt.clf()



if __name__=='__main__':

dir_in = "/Users/lucasso/Dropbox/UFMG/Processamento de Linguagem Natural/random_pck/docs/"
dir_parl = "/Users/lucasso/Documents/pck/"
dir_out = "/Users/lucasso/Dropbox/UFMG/Processamento de Linguagem Natural/"
file_parl = "/Users/lucasso/Dropbox/UFMG/Processamento de Linguagem Natural/random_pck/docs/deputados.pck"
tfidf_n = list()
tf_log_idf = list()
tfidf_like = list() 
corr = ""

with open(file_parl, 'rb') as handle:
    parl_counter = pickle.load(handle)

tot_counter,counter_list = loadCounters(dir_in)
tot_counter_dep,counter_list_dep= loadCounters(dir_parl)
tfidf = TfIdf()
for word in parl_counter:
    tf = tfidf.tf(word, parl_counter)
    idf = tfidf.idf(word,counter_list)
    log_idf = tfidf.idf_smooth(word,counter_list)
    ent_idf = tfidf.idf_like(word,parl_counter, tot_counter, counter_list, counter_list_dep)
    tfidf_n.append(tf*idf)
    tf_log_idf.append(tf*log_idf)
    tfidf_like.append(tf*ent_idf)

dic_tfidf= list(zip(parl_counter.keys(), tfidf_n))
dic_tf_log_idf= list(zip(parl_counter.keys(), tf_log_idf))
dic_tfidf_like= list(zip(parl_counter.keys(), tfidf_like))

"""
corr +=  "tfidf X tfidf_smooth: "+str(stats.spearmanr([v for i,v in dic_tfidf] ,[v for i,v in dic_tf_log_idf]))+"\n"
corr +=  "tfidf X tfidf_like: "+str(stats.spearmanr([v for i,v in dic_tfidf],[v for i,v in dic_tfidf_like]))+"\n"
corr +=  "tfidf_like X tfidf_smooth: "+str(stats.spearmanr([v for i,v in dic_tfidf_like] , [v for i,v in dic_tf_log_idf]))+"\n"

    corr +=  "tfidf X tfidf_smooth: "+str(stats.pearsonr(tfidf_n,tf_log_idf))+"\n"
    corr +=  "tfidf X tfidf_like: "+str(stats.pearsonr(tfidf_n,tfidf_like))+"\n"
    corr +=  "tfidf_like X tfidf_smooth: "+str(stats.pearsonr(tfidf_like,tf_log_idf))+"\n"

with open(dir_out+"saida_correlacao.txt", "w") as f:
    f.write(corr)
    f.close()

with open(dir_in+"dic_tfidf_like.pck", 'rb') as handle:
    dic_tfidf_like = pickle.load(handle)

with open(dir_out+"dic_tfidf.pck", 'wb') as handle:
    pickle.dump(dic_tfidf, handle)



"""

sort_tfidf = sorted(dic_tfidf, key=lambda x: x[1], reverse=True)
sort_tf_log_idf = sorted(dic_tf_log_idf, key=lambda x: x[1], reverse=True)
sort_tfidf_like = sorted(dic_tfidf_like, key=lambda x: x[1], reverse=True)

plot_tfidfs(sort_tfidf, sort_tf_log_idf, sort_tfidf_like)

n = 2000
plot_cloud(sort_tfidf,n,"dic_tfidf")
plot_cloud(sort_tf_log_idf,n,"dic_tf_log_idf")
plot_cloud(sort_tfidf_like,n,"dic_tfidf_like")

dir_path = "/Users/lucasso/Documents/tweets/"
tp = TextProcessor()
tw_files = ([file for root, dirs, files in os.walk(dir_path)
            for file in files if file.endswith('.json') ])

tw_list = list()
for tw_file in tw_files:
    with open(dir_path+tw_file) as data_file:
        doc_list = list()
        for line in data_file:
            tweet = json.loads(line)
            doc_list.append(tweet['text'])
    tw_list.append(list(itertools.chain.from_iterable(tp.text_process(doc_list))))

for i in range(len(tw_list)):
    plot_dep_cloud(tw_list[i],sort_tfidf,n,tw_files[i]+"_dic_tfidf")
    plot_dep_cloud(tw_list[i],sort_tf_log_idf,n,tw_files[i]+"dep_dic_tf_log_idf")
    plot_dep_cloud(tw_list[i],sort_tfidf_like,n,tw_files[i]+"dic_tfidf_like")






    


       