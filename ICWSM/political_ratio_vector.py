import sys
sys.path.append('../')
import argparse
import configparser
import numpy as np
from matplotlib import pyplot as plt
import pymongo
from political_classification import PoliticalClassification
import math
import random
from collections import defaultdict
from beautifultable import BeautifulTable
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_tsne(dictionary, filename):
    map_l = {'novos': 'Elected', 'reeleitos': 'Reelected',
             'nao_eleitos': 'Not Elected'}
    map_c = {'novos': 'orange', 'reeleitos': 'green',
             'nao_eleitos': 'purple'}
    tsne = TSNE(n_components=2, random_state=0, perplexity=10)
    figure = plt.figure(figsize=(15, 8))
    ax = figure.add_subplot(111)  
    for cond, values in dictionary.items():
        Y = tsne.fit_transform(values)
        ax.scatter(Y[:, 0], Y[:, 1], color=map_c[cond], label=map_l[cond])
    ax.set_title("T-SNE Relative Post Frequency Similarity")
    ax.legend()
    plt.savefig(dir_out + filename)

def plot_pca(dictionary, filename):
    map_l = {'novos': 'Elected', 'reeleitos': 'Reelected',
             'nao_eleitos': 'Not Elected'}
    map_c = {'novos': 'orange', 'reeleitos': 'green',
             'nao_eleitos': 'purple'}
    pca = PCA(n_components=2)
    figure = plt.figure(figsize=(15, 8))
    ax = figure.add_subplot(111)  
    for cond, values in dictionary.items():
        Y = pca.fit_transform(values)
        ax.scatter(Y[:, 0], Y[:, 1], color=map_c[cond], label=map_l[cond])
    ax.set_title("PCA Relative Post Frequency Similarity")
    ax.legend()
    plt.savefig(dir_out + filename)

if __name__ == "__main__":
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_out = path['dir_out']

    # election
    p1 = (1396483200000, 1412294400000)
    p2 = (1412294400000, 1443830400000)
    # impeachment
    p3 = (1459382400000, 1472601600000)
    p4 = (1472601600000, 1490918400000)

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb

    pc = PoliticalClassification('model_lstm.h5',
                            'dict_lstm.npy', 16)   
    
    periods = [p1, p2, p3, p4]
    politics = dict({'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    non_politics = dict({'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    
    for period in periods:
        print('getting data')
        #tweets = db.tweets.find({'created_at': {'$gte': period[0], '$lt': period[1]},
        #                          'cond_55': {'$exists': True}})
        tw = db.tweets.aggregate([{'$sample': {'size': 50000}},
                                      {'$match': {'created_at': {'$gte': period[0], '$lt': period[1]},
                                                  'cond_55': {'$exists': True}}}], allowDiskUse=True)
        print('processing tweets')
        for tweet in tw:
            if pc.is_political(tweet['text_processed']):
                if tweet['user_id'] not in politics[tweet['cond_55']]:
                    politics[tweet['cond_55']][tweet['user_id']] = defaultdict(int)
                politics[tweet['cond_55']][tweet['user_id']][period[0]] += 1
            else:
                if tweet['user_id'] not in non_politics[tweet['cond_55']]:
                    non_politics[tweet['cond_55']][tweet['user_id']] = defaultdict(int)
                non_politics[tweet['cond_55']][tweet['user_id']][period[0]] += 1

    print('creating vectors')
    f_pol = dict({'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})
    for cond, dep in politics.items():
        for _, val in dep.items():
            vec = [val[p1[0]], val[p2[0]], val[p3[0]], val[p4[0]]]
            vec = vec / np.sum(vec)
            f_pol[cond].append(vec)

    plot_tsne(f_pol, 'tsne_political.png')
    plot_pca(f_pol, 'pca_political.png') 

    f_pol = dict({'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})
    for cond, dep in non_politics.items():
        for _, val in dep.items():
            vec = [val[p1[0]], val[p2[0]], val[p3[0]], val[p4[0]]]
            vec = vec / np.sum(vec)
            f_pol[cond].append(vec)

    plot_tsne(f_pol, 'tsne_non_political.png')
    plot_pca(f_pol, 'pca_non_political.png') 




