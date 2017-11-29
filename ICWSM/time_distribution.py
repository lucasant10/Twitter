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
from collections import Counter


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

    if deputy is not None:
        ax.set_title("%s - Tweets Distribution Over Time" % deputy)

    else:
        ax.set_title("Tweets Distribution Over Time")
    plt.show()
    plt.clf()


def plot_hist(dist, title):
    num_bins = 50
    plt.hist(dist, num_bins)
    plt.title("%s Tweets Histogram" % title)
    plt.xlabel("Percent of %s" % title)
    plt.ylabel("Number of Deputies")
    plt.show()


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

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb

    if deputy is not None:
        tweets = db.tweets.find(
            {'created_at': {'$gte': 1380585600000, '$lt': 1506816000000},
             'user_name': deputy})
    elif condition is not None:
        tweets = db.tweets.find(
            {'created_at': {'$gte': 1380585600000, '$lt': 1506816000000},'cond_55': condition}).limit(10000)
    else:
        tweets = db.tweets.find(
            {'created_at': {'$gte': 1380585600000, '$lt': 1506816000000}}).limit(10000)

    pc = PoliticalClassification('model_lstm.h5', 'dict_lstm.npy', 18)

    p_count = {x: 0 for x in range(1, 54)}
    n_p_count = {x: 0 for x in range(1, 54)}
    p_count_dep = Counter()
    n_p_count_dep = Counter()
    p_party = dict()
    n_p_party = dict()
    print('processing tweets ...')
    for tweet in tweets:
        month = month_tw(int(tweet['created_at']))
        parl = tweet['user_name']
        
        if 'party' in tweet:
            party = tweet['party']
            if pc.is_political(tweet['text_processed']):
                p_count[month] += 1
                if party in p_party:
                    p_party[party][month] += 1
                else:
                    p_party[party] = Counter({x: 0 for x in range(1, 54)})
                    p_party[party][month] += 1

                if parl in p_count_dep:
                    p_count_dep[parl] += 1
                else:
                    p_count_dep[parl] = 1
            else:
                n_p_count[month] += 1
                if party in n_p_party:
                    n_p_party[party][month] += 1
                else:
                    n_p_party[party] = Counter({x: 0 for x in range(1, 54)})
                    n_p_party[party][month] += 1

                if parl in n_p_count_dep:
                    n_p_count_dep[parl] += 1
                else:
                    n_p_count_dep[parl] = 1

    print(p_count)
    print(n_p_count)
    if SAVE:
        cf = configparser.ConfigParser()
        cf.read("../file_path.properties")
        path = dict(cf.items("file_path"))
        dir_in = path['dir_in']

        np.save(dir_in + 'pol_counter_distribution.npy', p_count)
        np.save(dir_in + 'non_pol_counter_distribution.npy', n_p_count)
        with open(dir_in + 'samples_per_folder.txt', 'w') as out:
            out.write("\n\n political \n\n")
            out.write(prepare_text(p_count, 1000))
            out.write("\n\n non_political \n\n")
            out.write(prepare_text(n_p_count, 1000))

    print('plotting distribution ...')
    map_l = {'novos': 'Elected', 'reeleitos': 'Reelected',
             'nao_eleitos': 'Not Elected'}
    labels, p_values = zip(*sorted(p_count.items(), key=lambda i: i[0]))
    _, n_p_values = zip(*sorted(n_p_count.items(), key=lambda i: i[0]))

    dist_p = list()
    dist_n_p = list()
    for i, v in p_count_dep.items():
        dist_p.append(v / (v + n_p_count_dep[i]))
        dist_n_p.append(n_p_count_dep[i] / (v + n_p_count_dep[i]))

    if deputy is not None:
        plot_dist(p_values, n_p_values, labels, deputy)
    elif condition is not None:
        plot_dist(p_values, n_p_values, labels, None)
        plot_hist(dist_p, "%s Political" % map_l[condition])
        plot_hist(dist_n_p, "%s Non Political" % map_l[condition])
    else:
        plot_dist(p_values, n_p_values, labels, None)
        plot_hist(dist_p, "Political")
        plot_hist(dist_n_p, "Non Political")
        for party, counter in p_party.items():
            labels, p_values = zip(*sorted(counter.items(), key=lambda i: i[0]))
            _, n_p_values = zip(*sorted(n_p_party[party].items(), key=lambda i: i[0]))
            plot_dist(p_values, n_p_values, labels, party)
