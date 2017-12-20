import sys
sys.path.append('../')
import configparser
import numpy as np
from matplotlib import pyplot as plt
import pymongo
from political_classification import PoliticalClassification
from wordcloud import WordCloud
from collections import Counter, defaultdict
import math

def sem_tw(time):
    first = 1380585600000
    semester = 15778800000
    return math.ceil((time - first) / semester)


if __name__ == "__main__":
    
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_out = path['dir_out']

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    tweets = db.tweets.aggregate([ { '$sample': { 'size': 30000 }},
     { '$match': { 'created_at': {'$gte': 1380585600000, '$lt': 1506816000000},
     'cond_55': {'$exists': True} } } ])
	
    pc = PoliticalClassification('model_lstm.h5',
                            'dict_lstm.npy', 16)   
    parl = dict({'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    sem_tweets = defaultdict(str)
    for tweet in tweets:
        if pc.is_political(tweet['text_processed']):
            sem = sem_tw(tweet['created_at'])
            sem_tweets[sem] += tweet['text_processed']
            if sem not in parl[tweet['cond_55']]:
                parl[tweet['cond_55']][sem] = ''
            parl[tweet['cond_55']][sem] += tweet['text_processed']

    for s, txt in sem_tweets.items():
        wc = WordCloud().generate(txt)
        plt.imshow(wc)
        plt.savefig(dir_out + 'political_%s.png' % s ,dpi=100)

    for cond, s_tw in parl.items():
        for s, txt in s_tw.items():
            wc = WordCloud().generate(txt)
            plt.imshow(wc)
            plt.savefig(dir_out + 'political_%s_%s.png' % (cond, s) ,dpi=100)

