import nltk
from nltk.collocations import *
import os
import pickle
from text_processor import TextProcessor
import json
import pickle
import networkx as nx
import re




def load_file(dir_in, file_name):
    with open(dir_in+file_name, 'rb') as handle:
            fl = pickle.load(handle)
            return fl


def get_bigrams(tokens):
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    #bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)
    return bigram_finder

def load_files(dir_in):
    doc_list=list()
    tw_files = ([file for root, dirs, files in os.walk(dir_in)
        for file in files if file.endswith('.json') ])
    for tw_file in tw_files:
        with open(dir_in+tw_file) as data_file:
            for line in data_file:
                tweet = json.loads(line)
                doc_list.append(tweet['text'])           
    return doc_list



if __name__=='__main__':

    dir_in= "/Users/lucasso/Documents/pck/"
    dir_ent = "/Users/lucasso/Documents/tweets_pedro/"
    dir_out= "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/plot/"
    doc_list = load_files(dir_ent)
    tp = TextProcessor()
    tweets = tp.text_process(doc_list)
    word_list = set(load_file(dir_out,"word_list.pck"))
    #lista já processada sem entropia 0 e ration >1. remove todas as outras palavras que não interessam dos tweets
    tweets =[[i for i in t if i in word_list] for t in tweets] 
    hashtags = re.compile(r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""")
    hs_set =set()

    hastgs_list = list()
    for tweet in tweets:
        v = ','.join(hashtags.findall( ' '.join(tweet)))
        l = hashtags.findall( ' '.join(tweet))
        hastgs_list.append(l)
        hs_set |= set(v.split(","))

    hastgs_list = [e for e in hastgs_list if e] # remove as listas em branco     
    
    tweets =[[i for i in t if i not in hs_set] for t in tweets] # remove as hashtags dos tweets 

    #with open(dir_out+"hashtag_set.pck", 'wb') as handle:
    #    pickle.dump(hs_set, handle)
    #with open(dir_out+"hashtag_list.pck", 'wb') as handle:
    #    pickle.dump(hastgs_list, handle)
    #with open(dir_out+"tweeets_n_hstgs.pck", 'wb') as handle:
    #    pickle.dump(tweets, handle)

    #processa o grafo de hashtags

    graph = nx.DiGraph()
    for hashtags in hastgs_list:
        for i in range(len(hashtags)-1):
            first = hashtags[i]
            second = hashtags[i+1]
            if graph.has_edge(first,second):
                weight = graph.get_edge_data(first,second)['weight'] +1
                graph.edge[first][second]['weight']=weight
            else:
                graph.add_edge(first,second,weight=1)

    nx.write_gml(graph,dir_out+"grafo_hastags.gml")


    #processa o grafo de palavras

    graph = nx.DiGraph()
    for tweet in tweets:
        for i in range(len(tweet)-1):
            first = tweet[i]
            second = tweet[i+1]
            if graph.has_edge(first,second):
                weight = graph.get_edge_data(first,second)['weight'] +1
                graph.edge[first][second]['weight']=weight
            else:
                graph.add_edge(first,second,weight=1)

    nx.write_gml(graph,dir_out+"grafo_direcionado2.gml")
