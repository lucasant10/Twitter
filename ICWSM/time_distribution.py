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
from collections import Counter, defaultdict
import matplotlib.patches as mpatches


def month_tw(time):
    first = 1380585600000
    month = 2419200000
    return math.ceil((time - first) / month)


def prepare_text(counter, samples):
    lista = list(sorted(counter.items(), key=lambda i: i[0]))
    arr = np.asarray([x[1] for x in lista])
    arr = arr / sum(arr) * samples
    txt = ''
    for i, v in enumerate(arr):
        txt += 'month %s : %s \n' % ((i + 1), int(v))
    return txt


def plot_dist(p_values, n_p_values, labels, deputy):
    figure = plt.figure(figsize=(15, 8))
    ind = np.arange(1, (len(labels) + 1))
    width = 0.45
    ax = figure.add_subplot(111)
    p1 = ax.bar(ind, p_values, width, color='seagreen')
    p2 = ax.bar(ind + width, n_p_values, width, color='tomato')
    ax.set_xticks(ind)
    ax.legend((p1[0], p2[0]), ('Politcal', 'Non Political'))
    ax.set_xlabel('Months from oct/13 to oct/17')
    ax.set_ylabel('Number of tweets')
    ax.set_title("%s  Distribution Over Time" % deputy)
    plt.savefig(dir_in + "%s_distribution.png" % deputy)
    plt.clf()


def plot_hist(dist, title):
    num_bins = 20
    plt.hist(dist, num_bins)
    plt.title("%s Tweets Histogram" % title)
    plt.xlabel("Percentage of %s" % title)
    plt.ylabel("Number of Deputies")
    plt.savefig(dir_in + "%s_histogram.png" % title)
    plt.clf()


def plot_cdf(dist):
    figure = plt.figure(figsize=(15, 8))
    num_bins = 100
    ax = figure.add_subplot(111)
    for cond, values in dist.items():
        ax.hist(values, num_bins, normed=True, label=cond,
                cumulative=True, histtype='step')
    ax.legend(loc='upper left')
    ax.set_xlabel('Percentage of political tweets')
    ax.set_ylabel('Percentage of deputies')
    ax.set_title("Cumulatative Distributuion")
    plt.savefig(dir_in + "cdf_tweets.png")
    plt.clf()

def plot_disp(dist_tw, dist_p, colors):
    figure = plt.figure(figsize=(15, 8))
    ax = figure.add_subplot(111)
    ax.scatter(dist_tw, dist_p, color=colors)
    o_patch = mpatches.Patch(color='orange', label='Elected')
    g_patch = mpatches.Patch(color='green', label='Reelected')
    p_patch = mpatches.Patch(color='purple', label='Not Elected')
    ax.legend(handles=[o_patch, g_patch, p_patch])
    ax.set_xlabel('Number of tweets')
    ax.set_xscale('log')
    ax.set_ylabel('Percentage of political posts')
    ax.set_title("Proportion of political tweets by number of tweets")
    plt.savefig(dir_in + "disp_tweets.png")
    plt.clf()

def plot_dist_cdf(p_values, n_p_values, deputy):
    figure = plt.figure(figsize=(15, 8))
    ind = np.arange(1, (len(p_values) + 1))
    ax = figure.add_subplot(111)
    cum_y = np.cumsum(n_p_values)
    ax.plot(ind, np.cumsum(p_values), label='Political', color='seagreen')
    ax.plot(ind, cum_y, label='Non Political', color='tomato')
    ax.legend()
    ax.set_xticks(ind)
    ax.set_xlabel('Months from oct/13 to oct/17')
    ax.set_ylabel('Percentage of tweets')
    ax.set_title("%s Cumulative Distribution Over Time" % deputy)
    ax.axvline(14, color='b')
    ax.axvline(39, color='b')
    ax.annotate('election', 
             xy=(14,0.9),  
             xycoords='data',
             textcoords='offset points',
             arrowprops=dict(arrowstyle="->"))
    ax.annotate('impeachment', 
             xy=(39,0.9),  
             xycoords='data',
             textcoords='offset points',
             arrowprops=dict(arrowstyle="->"))
    plt.savefig(dir_in + "%s_cdf_distribution.png" % deputy)
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot histogram of tweets over time')
    parser.add_argument('--save',
                        action='store_true', default=False)
    parser.add_argument('-d', '--deputy')
    parser.add_argument('-c', '--condition')

    args = parser.parse_args()
    SAVE = args.save
    deputy = args.deputy
    condition = args.condition

    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_out']

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb

    if deputy is not None:
        tweets = db.tweets.find(
            {'created_at': {'$gte': 1380585600000, '$lt': 1506816000000},
             'user_name': deputy})
    elif condition is not None:
        tweets = db.tweets.find(
            {'created_at': {'$gte': 1380585600000, '$lt': 1506816000000}, 'cond_55': condition})
    else:
        tweets = db.tweets.find({'created_at': {'$gte': 1380585600000, '$lt': 1506816000000}, 'cond_55': {'$exists': True} }).limit(1000000)
        #tweets = db.tweets.aggregate([ { '$sample': { 'size': 400000 }}, { '$match': { 'created_at': {'$gte': 1380585600000, '$lt': 1506816000000}, 'cond_55': {'$exists': True} } } ], allowDiskUse=True)

    pc = PoliticalClassification('model_lstm.h5', 'dict_lstm.npy', 18)

    p_count = {x: 0 for x in range(1, 54)}
    n_p_count = {x: 0 for x in range(1, 54)}
    p_count_dep = defaultdict(int)
    n_p_count_dep = defaultdict(int)
    dep_set_dict = defaultdict(set)
    cond_set_dict = defaultdict(set)
    p_party = dict()
    n_p_party = dict()
    colors = list()
    map_l = {'novos': 'Elected', 'reeleitos': 'Reelected',
             'nao_eleitos': 'Not Elected'}
    map_c = {'novos': 'orange', 'reeleitos': 'green',
             'nao_eleitos': 'purple'}
    print('processing tweets ...')
    for tweet in tweets:
        month = month_tw(int(tweet['created_at']))
        parl = tweet['user_name']
        cond = tweet['cond_55']
        if 'party' in tweet:
            dep_set_dict[cond].add(parl)
            party = tweet['party']
            if pc.is_political(tweet['text_processed']):
                p_count[month] += 1
                if party in p_party:
                    p_party[party][month] += 1
                else:
                    p_party[party] = Counter({x: 0 for x in range(1, 54)})
                    p_party[party][month] += 1

                colors.append(map_c[cond])
                p_count_dep[parl] += 1

            else:
                n_p_count[month] += 1
                if party in n_p_party:
                    n_p_party[party][month] += 1
                else:
                    n_p_party[party] = Counter({x: 0 for x in range(1, 54)})
                    n_p_party[party][month] += 1

                n_p_count_dep[parl] += 1

    print(p_count)
    print(n_p_count)
    if SAVE:

        np.save(dir_in + 'pol_counter_distribution.npy', p_count)
        np.save(dir_in + 'non_pol_counter_distribution.npy', n_p_count)
        with open(dir_in + 'samples_per_folder.txt', 'w') as out:
            out.write("\n\n political \n\n")
            out.write(prepare_text(p_count, 1000))
            out.write("\n\n non_political \n\n")
            out.write(prepare_text(n_p_count, 1000))

    print('plotting distribution ...')
    labels, p_values = zip(*sorted(p_count.items(), key=lambda i: i[0]))
    _, n_p_values = zip(*sorted(n_p_count.items(), key=lambda i: i[0]))

    dist_p = list()
    dist_tw = list()
    for i, v in p_count_dep.items():
        dist_p.append(v / (v + n_p_count_dep[i]))
        dist_tw.append(v + n_p_count_dep[i])

    dist_cond = dict()
    print(dep_set_dict)
    print(p_count_dep)
    for cond, dep_set in dep_set_dict.items():
        p_temp = list()
        for dep in dep_set:
            count = p_count_dep[dep]
            p_temp.append(count / (count + n_p_count_dep[dep]))
        dist_cond[cond] = p_temp

    if deputy is not None:
        plot_dist(p_values, n_p_values, labels, deputy)
        plot_dist_cdf(p_values, n_p_values, labels, deputy)

    elif condition is not None:
        plot_dist(p_values, n_p_values, labels, "")
        plot_dist_cdf(p_values, n_p_values, "")
        plot_hist(dist_p, "%s Political" % map_l[condition])
    else:
        plot_dist(p_values, n_p_values, labels, "")
        plot_dist_cdf(p_values, n_p_values, "")
        plot_hist(dist_p, "Political")
        plot_cdf(dist_cond)
        plot_disp(dist_tw, dist_p, colors)
        for party, counter in p_party.items():
            labels, p_values = zip(
                *sorted(counter.items(), key=lambda i: i[0]))
            _, n_p_values = zip(
                *sorted(n_p_party[party].items(), key=lambda i: i[0]))
            plot_dist(p_values, n_p_values, labels, party)
