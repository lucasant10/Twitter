import sys
sys.path.append('../')
import argparse
import configparser
import json
import os
import math
import codecs
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import pymongo
from political_classification import PoliticalClassification


def month_tw(time):
    first = 1380585600000
    month = 2419200000
    return math.ceil((time - first)/month)


def prepare_text(counter, samples):
    l = list(sorted(count.items(), key=lambda i: i[0]))
    arr = np.asarray([x[1] for x in l])
    arr = arr/sum(arr) * samples
    txt = ''
    for i, v in enumerate(arr):
        txt += 'month %s : %s \n' % ((i+1), int(v))
    return txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot histogram of tweets over time')
    parser.add_argument('--save',
                        action='store_true', default=False)
    parser.add_argument('-d', '--deputy')

    args = parser.parse_args()
    SAVE = args.save
    deputy = args.deputy

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    tweets = db.tweets.find(
        {'created_at': {'$gte': 1380585600000, '$lt': 1506816000000}})

    pc = PoliticalClassification('model_lstm.h5', 'dict_lstm.npy', 18)

    p_count = Counter()
    n_p_count = Counter()
    print('processing tweets ...')
    for tweet in tweets:
        month = month_tw(int(tweet['created_at']))
        if pc.is_political(tweet['text_processed']):
            p_count[month] += 1
        else:
            n_p_count[month] += 1

    if SAVE:
        cf = configparser.ConfigParser()
        cf.read("../file_path.properties")
        path = dict(cf.items("file_path"))
        dir_in = path['dir_in']

        np.save(dir_in+'pol_counter_distribution.npy', p_count)
        np.save(dir_in+'non_pol_counter_distribution.npy', n_p_count)
        with open(dir_in + 'samples_per_folder.txt', 'w') as out:
            out.write("\n\n political \n\n")
            out.write(prepare_text(p_count, 1000))
            out.write("\n\n non_political \n\n")
            out.write(prepare_text(n_p_count, 1000))

    print('plotting distribution ...')
    labels, p_values = zip(*sorted(p_count.items(), key=lambda i: i[0]))
    _, n_p_values = zip(*sorted(n_p_count.items(), key=lambda i: i[0]))

    figure = plt.figure(figsize=(15, 8))
    plt.hist([
        p_values,
        n_p_values
    ],
        stacked=True, color=['b', 'r'],
        bins=len(labels), label=['Politcal, Non Political'])
    plt.xlabel('Months since oct/13 until oct/17')
    plt.ylabel('Number of tweets')
    plt.legend()
