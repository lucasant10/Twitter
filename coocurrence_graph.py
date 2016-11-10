import nltk
from nltk.collocations import *
import os
import pickle
import numpy as np
from text_processor import TextProcessor
import json
import itertools
import pickle
from word_table import WordTable
import networkx as nx
import re
from sklearn.feature_extraction.text import CountVectorizer



def load_word_list(dir_in, file_name):
    with open(dir_in+file_name, 'rb') as handle:
            word_list = pickle.load(handle)
            return word_list


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

    dir_in= "/home/lucasso/Documents/pck/"
    dir_ent = "/home/lucasso/Documents/tweets_pedro/"
    dir_out= "/home/lucasso/Documents/"
    doc_list = load_files(dir_ent)
    tp = TextProcessor()
    tweets = tp.text_process(doc_list)
    word_list = set(load_word_list(dir_out,"word_list.pck"))
    #lista já processada sem entropia 0 e ration >1. remove todas as outras palavras que não interessam dos tweets
    tweets =[[i for i in t if i in word_list] for t in tweets] 
    hashtags = re.compile(r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""")
    hs_set =set()

    hastgs_list = list()
    hastgs_list2 = list()
    for tweet in tweets:
        hastgs_list.append( ','.join(hashtags.findall( ' '.join(tweet))))
        hs_set |= set(v.split(","))

    hastgs_list = [e for e in hastgs_list if e] # remove as listas em branco
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(hashtags)
#processa o grafo

    graph = nx.DiGraph()
    for tweet in tw:
        for i in range(len(tweet)-1):
            first = tweet[i]
            second = tweet[i+1]
            if graph.has_edge(first,second):
                weight = graph.get_edge_data(first,second)['weight'] +1
                graph.edge[first][second]['weight']=weight
            else:
                graph.add_edge(first,second,weight=1)

    nx.write_gml(graph,dir_out+"grafo_direcionado2.gml")
    pos=nx.spring_layout(graph)
    weights = nx.get_edge_attributes(graph,'weight')
    nx.draw_networkx_nodes(graph,pos,node_size=150)
    nx.draw_networkx_edge_labels(graph,pos,edge_labels=weights)
    nx.draw_networkx_edges(graph,pos)
    plt.show()