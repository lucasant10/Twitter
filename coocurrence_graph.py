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

    dir_in= "/Users/lucasso/Documents/pck/"
    dir_ent = "/home/lucasso/Documents/"
    doc_list = load_files(dir_in)
    tp = TextProcessor()
    tweets = tp.text_process(doc_list)
    word_list = load_word_list(dir_ent,"word_list.pck")
    tweets =[[i for i in t if i in word_list] for t in tweets]
    hashtags = re.compile(r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""")
    hs_set =set()
    for tweet in tweets:
        hs_set |= set(hashtags.findall( ' '.join(tweet)))

    for tweet in tweets:



    graph = nx.DiGraph()

    for tweet in tweets:
        for i in range(len(tweet)-1)
            first = tweet[i]
            second = tweet[i+1]
            if graph.has_edge(first,second):
                weight = g.get_edge_data(first,second)['weight'] +1
                graph.edge[first][second]['weight']=weight
            else
                graph.add_edge(first,second,weight=1)

    nx.write_gml(s_lamb,dir_out+"grafo_direcionado.gml")