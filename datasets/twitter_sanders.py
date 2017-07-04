# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
from text_processor import TextProcessor
import configparser
import pickle
import numpy as np
import csv
import random

if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_btm = path['dir_btm']
    dir_in = path['dir_in']
    dir_out = path['dir_out']
    dir_down = path['dir_down']
    tp = TextProcessor()

    f = open(dir_down + "sanders-twitter-0.2/full-corpus.csv", "rt")
    twitter = csv.reader(f, delimiter=',')

    tweets = list()
    for tw in twitter:
        tweets.append(tw)

    random.shuffle(tweets)

    topic = list()
    txt = list()
    for tw in tweets:
        topic.append(tw[0])
        txt.append(tw[4])

    txt = tp.text_process(txt, lang="english")

    with open(dir_out + "sanders_twitter.pck", 'wb') as handle:
        pickle.dump(txt, handle)

    f = open(dir_out + "sanders_twitter.txt", 'w')
    for l in txt:
        f.write(" ".join(l) + "\n")

    f.close

    f = open(dir_out + "topic_sanders_twitter.txt", 'w')
    for t in topic:
        f.write(t + "\n")

    f.close
