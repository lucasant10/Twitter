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
from beautifultable import BeautifulTable
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def week_tw(time):
    first = 1391385600000
    week= 604800000
    return math.ceil((time - first)/week)

def dist_dep(val):
    bef = np.zeros(17)
    elec = np.zeros(17)
    for i in range(17):
        bef[i] += val[i]
        elec[i] += val[i+17]
    return bef, elec

def plot_cdf(b, e, filename):
    num_bins = 100
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

def category(val):
    if val < 0.001:
        return '0.001'
    elif val < 0.01:
        return '0.01'
    elif val < 0.03:
        return '0.03'
    else:
        return '0.05'



def distribution_process(distribution, dist_class):
    dist_plot = defaultdict(int)
    more_in_election = defaultdict(int)
    category = dict({'nao_eleitos': defaultdict(int), 'reeleitos': defaultdict(int), 'novos': defaultdict(int)})
    total = defaultdict(int)
    for cond, dep in distribution.items():
        total[cond] += 1
        for d, val in dep.items():
            before, election = dist_dep(val)
            ks = ks_2samp(before, election)
            # reject null hypotesis
            if ks[1] < 0.05:
                dist_plot[cond] += 1
                #plot_cdf(before, election, '%s_%s_%s.png' % (dist_class, cond, d))
                if np.mean(before) < np.mean(election):
                    category[cond][category(ks[1])] += 1
    print(dist_class)
    table = BeautifulTable()
    table.column_headers = ["", "0.001", "0.01", "0.03", "0.05"]
    table.append_row("reelected", category["reeleitos"]["0.001"], category["reeleitos"]["0.01"], category["reeleitos"]["0.03"], category["reeleitos"]["0.05"])
    table.append_row("not_elected", category["nao_eleitos"]["0.001"], category["nao_eleitos"]["0.01"], category["nao_eleitos"]["0.03"], category["nao_eleitos"]["0.05"])
    table.append_row("newcomer", category["novos"]["0.001"], category["novos"]["0.01"], category["novos"]["0.03"], category["novos"]["0.05"])
    print(table)
    print("percentage election")
    print("reelected %0.2f, not_elected %0.2f,newcomer %0.2f" % ((dist_plot['reelected'] / total['reelected']), (dist_plot['nao_eleitos'] / total['nao_eleitos']),(dist_plot['novos'] / total['novos'])))

if __name__ == "__main__":
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']

    # election
    p1 = (1391385600000, 1401753600000)
    p2 = (1401753600000, 1412294400000)
    
    pc = PoliticalClassification('cnn_s300.h5', 'cnn_s300.npy', 18)

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb

    periods = [p1, p2]
    politics = dict(
        {'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    non_politics = dict(
        {'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    both = dict(
        {'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    for period in periods:
        print('getting data')
        tweets = db.tweets.find({'created_at': {'$gte': period[0], '$lt': period[1]},
                                 'cond_55': {'$exists': True}})
        print('processing tweets')
        for tweet in tweets:

            if tweet['user_id'] not in both[tweet['cond_55']]:
                both[tweet['cond_55']][tweet['user_id']] = defaultdict(int)
            both[tweet['cond_55']][tweet['user_id']][week_tw(tweet['created_at'])] += 1

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

    distribution_process(both, 'both')
    distribution_process(politics, 'politics')
    distribution_process(non_politics, 'non-politics')
