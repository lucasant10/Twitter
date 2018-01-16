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
from sklearn.manifold import TSNE



def plot_cdf(dist, filename):
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
    plt.savefig(dir_in + filename)

def plot_freq(dictionary, filename):
    map_l = {'novos': 'Elected', 'reeleitos': 'Reelected',
             'nao_eleitos': 'Not Elected'}
    map_c = {'novos': 'orange', 'reeleitos': 'green',
             'nao_eleitos': 'purple'}
    figure = plt.figure(figsize=(15, 8))
    ax = figure.add_subplot(111)  
    tsne = TSNE(n_components=2, random_state=0, perplexity=30)
    for cond, values in dictionary.items():  
        Y = tsne.fit_transform(values)
        ax.scatter(Y[:, 0], Y[:, 1], color=map_c[cond], label=map_l[cond])
        # x = [i[0] for i in values]
        # y = [i[1] for i in values]
        # ax.scatter(x, y, color=map_c[cond], label=map_l[cond])
    ax.set_title("Relative Post Frequency Similarity")
    ax.legend()
    plt.savefig(dir_in + filename)
    figure.clf()

if __name__ == "__main__":
    
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_out']

    # # election
    # p1 = (1396483200000, 1412294400000)
    # p2 = (1412294400000, 1443830400000)
    # # impeachment
    # p3 = (1459382400000, 1472601600000)
    # p4 = (1472601600000, 1490918400000)

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    # tweets = db.tweets.find({'created_at': {'$gte': 1380585600000, '$lt': 1443830400000},
    #                          'cond_55': {'$exists': True}})
    tweets = db.tweets.aggregate([ { '$sample': { 'size': 100000 }}, 
        { '$match': { 'created_at': {'$gte': 1459382400000, '$lt': 1490918400000},
         'cond_55': {'$exists': True} } } ],  allowDiskUse=True)
    
    pc = PoliticalClassification('model_lstm.h5',
                            'dict_lstm.npy', 16)   

    politics = dict({'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    non_politics = dict({'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    for tweet in tweets:
        if pc.is_political(tweet['text_processed']):
            if tweet['user_id'] not in politics[tweet['cond_55']]:
                politics[tweet['cond_55']][tweet['user_id']] = defaultdict(int)
            if tweet['created_at'] <= 1472601600000:
                politics[tweet['cond_55']][tweet['user_id']]['before'] += 1
            else:        
                politics[tweet['cond_55']][tweet['user_id']]['after'] += 1
        else:
            if tweet['user_id'] not in non_politics[tweet['cond_55']]:
                non_politics[tweet['cond_55']][tweet['user_id']] = defaultdict(int)
            if tweet['created_at'] <= 1472601600000:
                non_politics[tweet['cond_55']][tweet['user_id']]['before'] += 1
            else:        
                non_politics[tweet['cond_55']][tweet['user_id']]['after'] += 1

    ratio = dict({'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})
    f_pol = dict({'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})
    for cond, dep in politics.items():
        for _, val in dep.items():
            r = (val['after'] + 1) / (val['before'] + 1) 
            ratio[cond].append(r)
            vec = [val['before'], val['after']]
            vec = vec / np.sum(vec)
            f_pol[cond].append(vec)

    plot_cdf(ratio, 'cdf_ratio_political.png')
    plot_freq(f_pol, 'tsne_ratio_political.png')

    table = BeautifulTable()
    table.column_headers = ["condition", "not_elected", "reelected", "elected"]
    table.append_row(["not_elected", 'X',
                    ks_2samp(ratio['nao_eleitos'], ratio['reeleitos'])[1],
                    ks_2samp(ratio['nao_eleitos'], ratio['novos'])[1]])
    table.append_row(["reelected", 'X','X', ks_2samp(ratio['reeleitos'], ratio['novos'])[1]])
    print(table)
    
    ratio = dict({'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})
    f_pol = dict({'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})
    for cond, dep in non_politics.items():
        for _, val in dep.items():
            r = (val['after'] + 1) / (val['before'] + 1) 
            ratio[cond].append(r)
            vec = [val['before'], val['after']]
            vec = vec / np.sum(vec)
            f_pol[cond].append(vec)
            
    plot_cdf(ratio, 'cdf_ratio_non_political.png')
    plot_freq(f_pol, 'tsne_ratio_non_political.png')

    table = BeautifulTable()
    table.column_headers = ["condition", "not_elected", "reelected", "elected"]
    table.append_row(["not_elected", 'X',
                    ks_2samp(ratio['nao_eleitos'], ratio['reeleitos'])[1],
                    ks_2samp(ratio['nao_eleitos'], ratio['novos'])[1]])
    table.append_row(["reelected", 'X','X', ks_2samp(ratio['reeleitos'], ratio['novos'])[1]])
    print(table)
    




