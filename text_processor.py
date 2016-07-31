# -*- coding: utf-8 -*-

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
#from stop_words import get_stop_words
from gensim import corpora, matutils
import gensim
import re
from read_twitter import ReadTwitter
from unicodedata import normalize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import numpy as np
import math




class TextProcessor:

    tokenizer = RegexpTokenizer(r'\w+')

    stoplist  = stopwords.words("portuguese")+['del','bom','via','nova','agora','boa','aqui', 'foto']
    #stoplist = get_stop_words('portuguese')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = SnowballStemmer("portuguese")

    # remvove urls

    def remove_urls(self, txt):  
        txt = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', txt)
        return txt


    def remover_acentos(self, txt):
        return normalize('NFKD', txt).encode('ASCII','ignore').decode('ASCII')

    def plot_text(self,texts, name, dir_out):
        txt = ""
        tokens = self.text_process(texts,False)
        for text in tokens :
            for word in text:
                txt += " "+word
        wc = WordCloud().generate(txt)
        plt.imshow(wc)
        plt.savefig(dir_out+name+'.png', dpi=300)

    def plot_text_stem(self,texts, name, dir_out):
        txt = ""
        tokens = self.text_process(texts)
        for text in tokens :
            for word in text:
                txt += " "+word
        wc = WordCloud().generate(txt)
        plt.imshow(wc)
        plt.savefig(dir_out+name+'.png', dpi=300)

    def text_process(self,doc_set, stem=True):
        # list for tokenized documents in loop
        texts = []
        # loop through document list
        for i in doc_set:
            # clean and tokenize document string
            raw = i.lower()

            #remove urls
            raw = self.remove_urls(raw) 
            #raw = remover_acentos(raw)
    
            tokens = self.tokenizer.tokenize(raw)
          
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in self.stoplist]

            #remove unigrams and bigrams
            stopped_tokens = [i for i in stopped_tokens if len(i) > 2]
            # remove acentos
            stopped_tokens = [self.remover_acentos(i) for i in stopped_tokens]
            
            if stem:
                # stem tokens
                stopped_tokens = [self.p_stemmer.stem(i) for i in stopped_tokens]
            
            # add tokens to list
            texts.append(stopped_tokens)
        return texts

    def create_corpus(self,texts):
        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(texts)
        dictionary.compactify()
        # and save the dictionary for future use
        #dictionary.save('tweet_teste.dict')

            
        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in texts]

        # and save in Market Matrix format
        #corpora.MmCorpus.serialize('tweet_teste.mm', corpus)
        # this corpus can be loaded with corpus = corpora.MmCorpus('tweet_teste.mm')
        return (corpus, dictionary)


    def  generate_lda(self, corpus, dictionary, num_topics):
            
        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary,alpha='auto')
        #ldamodel.save('tweet_teste.lda')
        #model = gensim.models.LdaModel.load('android.lda')
        print(ldamodel.print_topics())
        #ldamodel.print_topics()
        return ldamodel

    def  generate_hdp(self, corpus, dictionary):
            
        # generate LDA model
        hdpmodel = gensim.models.HdpModel(corpus, id2word = dictionary)
        #ldamodel.save('tweet_teste.lda')
        #model = gensim.models.LdaModel.load('android.lda')
        print(hdpmodel.print_topics(topics=-1, topn=20))
        #ldamodel.print_topics()
        return hdpmodel

    def print_topics(self, ldamodel, topn=10):
        
        Lambda = ldamodel.state.get_lambda()

        Phi = Lambda / Lambda.sum(axis=1)[:, np.newaxis]
        Phi2 =  Lambda / Lambda.sum(axis=0)[np.newaxis, :]
        entropy = np.zeros(Phi2.shape[1])
        topics = ""

        # calcula a entropia Ew≜∑kp(k|w)logp(k|w)
        for w in range(Phi2.shape[1]):
            for k in range(Phi2.shape[0]):
                entropy[w] += Phi2[k,w]*np.log2(Phi2[k,w]+1e-100)
        print(entropy)

        # calcula p(w|k)e−Hw
        for k in range(Phi.shape[0]):
            for w in range(Phi.shape[1]):
                Phi[k,w] = Phi[k,w]/pow(math.e,(-1)*entropy[w])
        for k in range(Phi.shape[0]):
            bestn = matutils.argsort(Phi[k], topn, reverse=True)
            topic_terms = [(id, Phi[k,id]) for id in bestn]
            lda_words = [(ldamodel.id2word[id], value) for id, value in topic_terms]    
            topics += ' + '.join(['%.3f*%s' % (v, k) for k, v in lda_words])+"\n"
        return topics


if __name__=='__main__':

    dir_in = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/"
    dir_out = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/plot/"
    excel_path = "/Users/lucasso/Dropbox/Twitter_Marcelo/Arquivo Principal da Pesquisa - Quatro Etapas.xls"
    sheet_name = "Sheet1"
    col = 4
    rt = ReadTwitter(dir_in, excel_path, "novos", col )
    rt2 = ReadTwitter(dir_in, excel_path, "reeleitos", col )
    rt3 = ReadTwitter(dir_in, excel_path, "nao_eleitos", col )
    tp = TextProcessor()
    #name, doc =  rt.tweets_by_rep()
    antes, depois = rt.tweets_before_after()
    antes = tp.text_process(antes)
    depois = tp.text_process(depois)
    corpus_antes, dic_antes  = tp.create_corpus(antes)
    corpus_depois, dic_depois = tp.create_corpus(depois)
    tp.generate_lda(corpus_antes, dic_antes, 15)
    tp.generate_lda(corpus_depois, dic_depois, 15)




"""
    for i in range(len(name)):
        tp.plot_text(doc[i],name[i])
        #doc_set = set()
        #doc_set = rt.tweets()
        
        filedir = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/74171_Chico Alencar.json"
        with open(filedir) as data_file:
            for line in data_file:
                tweet = json.loads(line)
                doc_set.add(tweet['text'])
        """

