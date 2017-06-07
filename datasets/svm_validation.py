import sys
sys.path.append('../')
import configparser
from text_processor import TextProcessor
import topic_BTM as btm
import numpy as np
import math
import csv
import os
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def return_files(extesion, path):
    fnames = ([file for root, dirs, files in os.walk(path)
               for file in files if file.endswith(extesion)])
    return fnames

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python <model_dir> <text_file> <topics_file>')
        exit(1)
    model_dir = sys.argv[1]
    text_file = sys.argv[2]
    text_topics = sys.argv[3]

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_dataset = path['dir_dataset']

    print("Reading topics distribution")
    voca = btm.read_voca(dir_dataset + model_dir + "voca.txt")
    inv_voca = {v: k for k, v in voca.items()}
    pz_pt = dir_dataset + model_dir + \
        return_files(".pz", dir_dataset + model_dir)[0]
    pz = btm.read_pz(pz_pt)
    zw_pt = dir_dataset + model_dir + \
        return_files(".pw_z", dir_dataset + model_dir)[0]

    print("Getting topics from files")
    k = 0
    topics = []
    vectors = np.zeros((len(pz), len(inv_voca)))
    for k, l in enumerate(open(zw_pt)):
        vs = [float(v) for v in l.split()]
        vectors[k, :] = vs
        wvs = zip(range(len(vs)), vs)
        wvs = sorted(wvs, key=lambda d: d[1], reverse=True)
        topics.append(wvs)
        k += 1

    print("Reading file %s" % text_file)
    texts = list()
    txt = open(dir_dataset + text_file, "r")
    for l in txt:
        texts.append(l.split())

    print("Assign topic to each text")
    assing_topics = list()
    dist_topics = list()
    for t in texts:
        tw_topics = list()
        lista = list()
        for tp in topics:
            tw = dict()
            tp = dict(tp)
            tmp = 0
            for w in t:
                if(w in inv_voca):
                    tmp += tp[inv_voca[w]]
            tw_topics.append(tmp)
            lista.append(tw)
        dist_topics.append(tw_topics)
        assing_topics.append(tw_topics.index(max(tw_topics)))


    print("Reading topics from file")
    labels = list()
    tw = open(dir_dataset + text_topics, "r")
    for l in tw:
        labels.append(l)

    i = 0
    num_topics = dict()
    for tp in set(labels):
        num_topics[tp] = i
        i = i + 1

    labels = [num_topics[x] for x in labels]
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, dist_topics, labels, cv=k-1)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(clf,dist_topics, labels, cv=k-1, scoring='f1_macro')
    print("F1_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



