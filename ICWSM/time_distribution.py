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

def month_tw(time):
    first = 1380585600000
    month= 2419200000
    return math.ceil((time - first)/month)

def load_files(dir_in):
    doc_list = list()
    tw_files = sorted([file for root, dirs, files in os.walk(dir_in)
                 for file in files if file.endswith('.json')])
    # 10/2013 1380585600
    # 10/2017 1506816000
    for tw_file in tw_files:
        temp = list()
        with open(dir_in+tw_file) as data_file:
            for line in data_file:
                tweet = json.loads(line)
                time = int(tweet['created_at'])
                if(1380585600000 <= time <= 1506816000000 ):
                    temp.append(tweet)
        doc_list.append(temp)
    return doc_list, tw_files

def save_file(path, filename, dic_file):
    for dirs, tweets in dic_file.items():
        os.makedirs(os.path.dirname(dirs), exist_ok=True)
        f = codecs.open(dirs + filename, 'a', 'utf-8')
        for tw in tweets:
            json.dump(tw, f)
        f.close()

def prepare_text(counter, samples):
    l = list(counter.items())
    arr = np.asarray([x[0] for x in l])
    arr = arr/sum(arr) * samples
    txt = ''
    for i, v in enumerate(num):
        txt += 'month %s : %s \n' %((i+1), int(v)) 
    return txt

if __name__ == "__main__":
    
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']

    count = Counter()
    tweet = list()
    doc_list, tw_files = load_files(dir_in)
    for i, tweet in enumerate(doc_list):
        temp = dict()
        for tw in tweet:
            month = month_tw(int(tw['created_at']))
            count[month] += 1
            path =  dir_in + str(month) + os.path.sep
            if path not in temp:
                temp[path] = [tw]
            else:
                temp[path].append(tw)
        save_file(path, tw_files[i], temp)
    
    np.save(dir_in+'tw_counter_distribution.npy', count) 
    with open(dir_in + 'samples_per_folder.txt', 'w') as out:
        out.write(prepare_text(count, 1000))

    labels, values = zip(*sorted(count.items(), key=lambda i: i[0]))
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()