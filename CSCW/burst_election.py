import matplotlib
matplotlib.use('Agg')
import sys
import os
sys.path.append('../')
import argparse
import configparser
import numpy as np
from matplotlib import pyplot as plt
import pymongo
from political_classification import PoliticalClassification
import math
import seaborn as sns
from scipy.stats import ks_2samp
from collections import defaultdict
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def week_tw(time):
    first = 1396483200000
    week= 604800000
    return math.ceil((time - first)/week)

def dist_dep(val):
    bef = np.zeros(14)
    elec = np.zeros(14)
    for i in range(14):
        bef[i] += val[i]
        elec[i] += val[i+14]
    return bef, elec

def plot_cdf(b, e, filename):
    num_bins = 14
    b_counts, b_bin_edges = np.histogram (b, bins=num_bins, normed=True)
    e_counts, e_bin_edges = np.histogram (e, bins=num_bins, normed=True)
    b_cdf = np.cumsum (b_counts)
    e_cdf = np.cumsum (e_counts)
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.plot (b_bin_edges[1:], b_cdf/b_cdf[-1], label= 'before election')
    ax.plot (e_bin_edges[1:], e_cdf/e_cdf[-1], label='election term')
    ax.legend(loc='upper left')
    # fig, ax = plt.subplots()
    # sns.kdeplot(b,ax=ax, cumulative=True, label='before election')
    # sns.kdeplot(e,ax=ax, cumulative=True, label='election term')
    # sns.distplot(x,
    #     hist_kws=dict(cumulative=True),
    #     kde_kws=dict(cumulative=True))
    plt.savefig(dir_in + "burst/" + filename)
    plt.clf()

    def plot_cdf(politics, non_politics, label):
    map_l = {'novos': 'New', 'reeleitos': 'Reelected', 'nao_eleitos': 'Loser'}
    num_bins = 100
    p_values = list()
    np_values = list()
    for cond, values in politics.items():
        p_values += values
        np_values += non_politics[cond]
    print(stats.ks_2samp(p_values,np_values))
    # p_values = np.where(np.isneginf(np.log10(p_values)), 0, np.log10(p_values))
    # np_values = np.where(np.isneginf(np.log10(np_values)), 0, np.log10(np_values))
    p_counts, p_bin_edges = np.histogram (p_values, bins=num_bins, normed=True)
    np_counts, np_bin_edges = np.histogram (np_values, bins=num_bins, normed=True)
    p_cdf = np.cumsum (p_counts)
    np_cdf = np.cumsum (np_counts)
    figure = plt.figure(figsize=(15, 8))
    ax = figure.add_subplot(111)
    ax.plot (p_bin_edges[1:], p_cdf/p_cdf[-1], label= 'political')
    ax.plot (np_bin_edges[1:], np_cdf/np_cdf[-1], label='non-political')
    ax.legend(loc='upper left')
    ax.set_xlabel('# of %s tweets'% str.lower(label))
    ax.set_ylabel('F(x)')
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    #ax.set_xscale("log", basex=2)
    #ax.set_yticks(np.arange(0.5, 1.0, step=0.1))
    ax.set_xticks(np.arange(0, 50, step=1))
    plt.xlim(xmax=50)
    plt.xlim(xmin=0)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(dir_in + 'cdf_%s.png' % (label))
    plt.clf()


def distribution_process(distribution, dist_class):
    dist_plot = defaultdict(int)
    more_in_election = defaultdict(int)
    total = defaultdict(int)
    for cond, dep in distribution.items():
        total[cond] += 1
        for d, val in dep.items():
            before, election = dist_dep(val)
            ks = ks_2samp(before, election)
            # reject null hypotesis
            if ks[1] > 0.05:
                dist_plot[cond] += 1
            else:
                if np.mean(before) < np.mean(election):
                    plot_cdf(before, election, '%s_%s_%s.png' % (dist_class, cond, d))
                    more_in_election[cond] += 1

    print("same distribution")
    print(dist_plot)
    print("more in election")
    print(more_in_election)


if __name__ == "__main__":
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']

    # election
    p1 = (1396483200000, 1404345600000)
    p2 = (1404345600000, 1412294400000)
    
    pc = PoliticalClassification('cnn_s300.h5', 'cnn_s300.npy', 18)

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb

    periods = [p1, p2]
    politics = dict(
        {'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    non_politics = dict(
        {'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    for period in periods:
        print('getting data')
        tweets = db.tweets.find({'created_at': {'$gte': period[0], '$lt': period[1]},
                                 'cond_55': {'$exists': True}})
        print('processing tweets')
        for tweet in tweets:
            if pc.is_political(tweet['text_processed']):
                if tweet['user_id'] not in politics[tweet['cond_55']]:
                    politics[tweet['cond_55']
                             ][tweet['user_id']] = defaultdict(int)
                politics[tweet['cond_55']][tweet['user_id']][week_tw(tweet['created_at'])] += 1
            else:
                if tweet['user_id'] not in non_politics[tweet['cond_55']]:
                    non_politics[tweet['cond_55']
                                 ][tweet['user_id']] = defaultdict(int)
                non_politics[tweet['cond_55']
                             ][tweet['user_id']][week_tw(tweet['created_at'])] += 1

    print('processing distributions')

    distribution_process(politics, 'politics')
    distribution_process(non_politics, 'non-politics')