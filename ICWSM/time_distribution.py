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

def month_tw(time):
    first = 1380585600000
    month= 2419200000
    return math.ceil((time - first)/month)

def save_file(path, filename, dic_file):
    for dirs, tweets in dic_file.items():
        os.makedirs(os.path.dirname(dirs), exist_ok=True)
        f = codecs.open(dirs + filename, 'a', 'utf-8')
        for tw in tweets:
            json.dump(tw, f)
        f.close()

def prepare_text(counter, samples):
    l = list(sorted(count.items(), key=lambda i: i[0]))
    arr = np.asarray([x[1] for x in l])
    arr = arr/sum(arr) * samples
    txt = ''
    for i, v in enumerate(arr):
        txt += 'month %s : %s \n' %((i+1), int(v)) 
    return txt


if __name__ == "__main__":
    
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    tweets = db.tweets.find({'created_at' : {'$gte':1380585600000, '$lt':1506816000000}})
    
    count = Counter()
    for tweet in tweets:
        month = month_tw(int(tweet['created_at']))
        count[month] += 1

    np.save(dir_in+'tw_counter_distribution.npy', count) 
    with open(dir_in + 'samples_per_folder.txt', 'w') as out:
        out.write(prepare_text(count, 1000))
    labels, values = zip(*sorted(count.items(), key=lambda i: i[0]))
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()
