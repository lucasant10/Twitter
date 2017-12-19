import sys
sys.path.append('../')
import argparse
import configparser
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import pymongo
from political_classification import PoliticalClassification
import math
import random
from collections import defaultdict
from scipy.stats import ks_2samp
from beautifultable import BeautifulTable



def plot_cdf(dist):
    map_l = {'novos': 'Elected', 'reeleitos': 'Reelected',
             'nao_eleitos': 'Not Elected'}
    figure = plt.figure(figsize=(15, 8))
    num_bins = 100
    ax = figure.add_subplot(111)
    for cond, values in dist.items():
        ax.hist(values, num_bins, normed=True, label=map_l[cond],
                histtype='step', cumulative=True)
    ax.legend(loc='upper left')
    ax.set_xlabel('Ratio of political tweets after/before election date')
    ax.set_xscale('log', basex=2), 
    ax.set_ylabel('Percentage of Parliamentarians')
    ax.set_title("Cumulatative Distributuion")
    plt.savefig(dir_in + "cdf_ratio.png")
    plt.clf()

if __name__ == "__main__":
    
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_out']

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    # tweets = db.tweets.find({'created_at': {'$gte': 1380585600000, '$lt': 1443830400000},
    #                          'cond_55': {'$exists': True}})
    tweets = db.tweets.aggregate([ { '$sample': { 'size': 30000 }}, 
        { '$match': { 'created_at': {'$gte': 1380585600000, '$lt': 1443830400000},
         'cond_55': {'$exists': True} } } ])
    
    pc = PoliticalClassification('model_lstm.h5',
                            'dict_lstm.npy', 16)   
    parl = dict({'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    for tweet in tweets:
        if pc.is_political(tweet['text_processed']):
            if tweet['user_id'] not in parl[tweet['cond_55']]:
                parl[tweet['cond_55']][tweet['user_id']] = defaultdict(int)
            if tweet['created_at'] <= 1412294400000:
                parl[tweet['cond_55']][tweet['user_id']]['before'] += 1
            else:        
                parl[tweet['cond_55']][tweet['user_id']]['after'] += 1

    ratio = dict({'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})
    for cond, dep in parl.items():
        for _, val in dep.items():
            r = (val['after'] + 1) / (val['before'] + 1) 
            ratio[cond].append(r)

    plot_cdf(ratio)

    table = BeautifulTable()
    table.column_headers = ["condition", "not_elected", "reelected", "elected"]
    table.append_row(["not_elected", 'X',
                    ks_2samp(ratio['nao_eleitos'], ratio['reeleitos'])[1],
                    ks_2samp(ratio['nao_eleitos'], ratio['novos'])[1]])
    table.append_row(["reelected", 'X','X', ks_2samp(ratio['reeleitos'], ratio['novos'])[1]])
    print(table)
    print(ks_2samp(ratio['nao_eleitos'], ratio['reeleitos'])[1])
    print(ks_2samp(ratio['nao_eleitos'], ratio['novos'])[1])
    print(ks_2samp(ratio['reeleitos'], ratio['novos'])[1])





