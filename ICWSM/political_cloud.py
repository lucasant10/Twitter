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
import random
from googletrans import Translator

def sem_tw(time):
    first = 1380585600000
    semester = 15778800000
    return math.ceil((time - first) / semester)

def freq_translated(counter):
    print('translating')
    translator = Translator()
    temp = counter.most_common(200)
    words = translator.translate([x[0] for x in temp], src='pt', dest='en')
    w_list = [w.text for w in words]
    w_freq = {x : temp[i][1] for i, x in enumerate(w_list)}
    return w_freq


def grey_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(0, 0%%, %d%%)" % 0

if __name__ == "__main__":
    
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_out = path['dir_out']

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    tweets = db.tweets.aggregate([ { '$sample': { 'size': 10000 }},
     { '$match': { 'created_at': {'$gte': 1380585600000, '$lt': 1506816000000},
     'cond_55': {'$exists': True} } } ], allowDiskUse=True)
	
    pc = PoliticalClassification('model_lstm.h5',
                            'dict_lstm.npy', 16)   

    politics = dict({'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    non_politics = dict({'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    pol_sem_tweets = defaultdict(Counter)
    non_pol_sem_tweets = defaultdict(Counter)
    print('processing tweets')
    for tweet in tweets:
        sem = sem_tw(tweet['created_at'])
        text = tweet['text_processed']
        for w in ['sobre', 'hoje']:
            text = text.replace(w, '')
        if pc.is_political(text):
            pol_sem_tweets[sem] += Counter(text.split(' '))
            if sem not in politics[tweet['cond_55']]:
                politics[tweet['cond_55']][sem] = Counter()
            politics[tweet['cond_55']][sem] += Counter(text.split(' '))
        else:
            non_pol_sem_tweets[sem] += Counter(text.split(' '))
            if sem not in non_politics[tweet['cond_55']]:
                non_politics[tweet['cond_55']][sem] = Counter()
            non_politics[tweet['cond_55']][sem] += Counter(text.split(' '))

    print('plotting')
    for s, counter in pol_sem_tweets.items():
        w_freq = freq_translated(counter)
        wc = WordCloud(background_color=None, max_font_size=50, mode='RGBA', color_func=grey_color_func).fit_words(w_freq)
        plt.imshow(wc, interpolation="bilinear")
        plt.savefig(dir_out + 'political_%s.png' % s ,dpi=300)
        plt.axis("off")
        plt.clf()

    for cond, s_tw in politics.items():
        for s, counter in s_tw.items():
            w_freq = freq_translated(counter)
            wc = WordCloud(background_color=None, max_font_size=50, mode='RGBA', color_func=grey_color_func).fit_words(w_freq)
            plt.imshow(wc, interpolation="bilinear")
            plt.savefig(dir_out + 'political_%s_%s.png' % (cond, s) ,dpi=300)
            plt.axis("off")
            plt.clf()

    for s, counter in non_pol_sem_tweets.items():
        w_freq = freq_translated(counter)
        wc = WordCloud(background_color=None, max_font_size=50, mode='RGBA', color_func=grey_color_func).fit_words(w_freq)
        plt.imshow(wc, interpolation="bilinear")
        plt.savefig(dir_out + 'non_political_%s.png' % s ,dpi=300)
        plt.axis("off")
        plt.clf()

    for cond, s_tw in non_politics.items():
        for s, counter in s_tw.items():
            w_freq = freq_translated(counter)
            wc = WordCloud(background_color=None, max_font_size=50, mode='RGBA', color_func=grey_color_func).fit_words(w_freq)
            plt.imshow(wc, interpolation="bilinear")
            plt.savefig(dir_out + 'non_political_%s_%s.png' % (cond, s) ,dpi=300)
            plt.axis("off")
            plt.clf()

