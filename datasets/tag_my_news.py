# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
from text_processor import TextProcessor
import configparser
import pickle
import numpy as np


def purity_score(clusters, classes):
    A = np.c_[(clusters, classes)]
    n_accurate = 0.
    for j in np.unique(A[:, 0]):
        z = A[A[:, 0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])
    return n_accurate / A.shape[0]

if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_btm = path['dir_btm']
    dir_in = path['dir_in']
    dir_out = path['dir_out']
    dir_down = path['dir_down']
    tp = TextProcessor()

tag_my_news = open(dir_down + "TagMyNews/en09062011.news", "r")
content = list()
for line in tag_my_news:
    tmp = line
    if(line[-1] == '\n'):
        content.append(tmp)
    else:
        tmp += line

txt = list()
title = list()
topic = list()
for i, v in enumerate(content):
    if(i % 8 == 0):
        title.append(v)
    elif(i % 8 == 1):
        txt.append(v)
    elif(i % 8 == 6):
        topic.append(v)

txt = tp.text_process(txt, lang="english")

    with open(dir_out + "tag_my_news.pck", 'wb') as handle:
        pickle.dump(txt, handle)

f = open(dir_out + "tag_my_news.txt", 'w')
for l in txt:
    f.write(" ".join(l) + "\n")

f.close

f = open(dir_out + "topic_tag_my_news.txt", 'w')
for t in topic:
    f.write(t)

f.close
# from sklearn.metrics.cluster import adjusted_rand_score
# adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0])
# from sklearn.metrics.cluster import normalized_mutual_info_score
# normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
