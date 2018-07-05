import sys
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
    plt.clf()
    sns.kdeplot(b, cumulative=True, label='before election')
    sns.kdeplot(e, cumulative=True, label='election term')
    # sns.distplot(x,
    #     hist_kws=dict(cumulative=True),
    #     kde_kws=dict(cumulative=True))
    plt.savefig(dir_in + filename)

def distribution_process(distribution, dist_class)
    dist_plot = defaultdict(int)
    total = defaultdict(int)
    for cond, dep in distribution.items():
        total[cond] += 1
        for d, val in dep.items():
            before, election = dist_dep(val)
            ks = ks_2samp(before, election)
            # reject null hypotesis
            if ks[1] <= 0.1:
                dist_plot[cond] += 1
            else:
                plot_cdf(before, election, '%s_%s_%d.png' % (dist_class, cond, d))
    print(dist_plot)


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
