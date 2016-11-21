import nltk
from nltk.collocations import *
import os
import pickle
import numpy as np
from text_processor import TextProcessor
import json
import itertools
import pickle



def load_words_entropy(dir_in, file_name):
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

	dir_in = "/home/lucasso/Documents/tweets_pedro/"
	dir_ent = "/home/lucasso/Documents/"

	doc_list = load_files(dir_in)

	tp = TextProcessor()
	tweets = tp.text_process(doc_list)
	tweets = list(itertools.chain.from_iterable(tweets))
	word_list = load_words_entropy(dir_ent,"lista_entr_zero.pck")
	#remover as palavras com entropia 0
	tweets = [i for i in tweets if i in word_list]
	bigram_finder = get_bigrams(tweets)
	bigram_measures = nltk.collocations.BigramAssocMeasures()
	print(bigram_finder.nbest(bigram_measures.pmi, 500))










