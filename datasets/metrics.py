import sys
sys.path.append('../')
import configparser
from text_processor import TextProcessor
import topic_BTM as btm
import numpy as np
import math
import csv
import os
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import completeness_score
from collections import Counter
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score


def return_files(extesion, path):
    fnames = ([file for root, dirs, files in os.walk(path)
               for file in files if file.endswith(extesion)])
    return fnames


def purity_score(clusters, classes):
    A = np.c_[(clusters, classes)]
    n_accurate = 0.
    for j in np.unique(A[:, 0]):
        z = A[A[:, 0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])
    return n_accurate / A.shape[0]


def coherence_value(words, docs, vocab):
    # Test if each word w and w+1 in a topic are present in doc
    # Better if the value is near by 0
    c_sum = 0
    for i in range(len(words) - 1):
        presence = 0
        presence_both = 0
        for doc in docs:
            if(vocab[words[i + 1]] in doc):
                presence += 1
                if((vocab[words[i]] in doc)):
                    presence_both += 1
        c_sum += math.log2((presence_both + 1) / presence)
    return c_sum


def topic_coherence(topics, docs, vocab, n):
    coherence_l = list()
    for tp in topics:
        n_words = [k for (k, v) in tp[:n]]
        coherence = coherence_value(n_words, docs, vocab)
        coherence_l.append(coherence)
    return coherence_l


def assign_topic_label(topics, labels):
    topic_set = set(topics)
    labels_set = set(labels)
    if(len(topic_set) != len(topic_set)):
        print("different number of topics, top: %d, labels %d",
              (len(topic_set), len(topic_set)))
        return
    topic_p = list()
    t_labels = list()
    for k in topic_set:
        topic_p.append(Counter([labels[i]
                                for i, v in enumerate(topics) if v == k]))
    while len(labels_set) != 0:
        top = list()
        for i, v in enumerate(topic_set):
            t = topic_p[i].most_common(1)
            if t == []:
                # last element without group
                t = [(list(labels_set)[0], 1)]
            top.append((v, t))
        top = sorted(top, key=lambda x: x[1][0][1], reverse=True)
        for i, x in enumerate(top):
            if(x[1][0][0] in labels_set):
                t_labels.append((x[0], x[1][0][0]))
                topic_set.remove(x[0])
                labels_set.remove(x[1][0][0])
                [i.__delitem__(x[1][0][0]) for i in topic_p]
                topic_p = topic_p[i:]
            else:
                break
    d_labels = dict(t_labels)
    return [d_labels[x] for x in topics]


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
    assign_topics = list()
    tw_l = list()
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
                    tw[w] = tp[inv_voca[w]]
                    tmp += tp[inv_voca[w]]
            tw_topics.append(tmp)
            lista.append(tw)
        dist_topics.append(tw_topics)
        tw_l.append(lista)
        assign_topics.append(tw_topics.index(max(tw_topics)))

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

    print("Saving %s_metrics file" % text_file)
    f = open(dir_dataset + "%s_metrics.txt" % text_file, 'w+')
    f.write("Metrics for %s\n" % text_file)
    f.write("Adjusted Rand Index(ARI)\n")
    f.write(str(adjusted_rand_score(assign_topics, labels)))
    f.write("\nNormalized Mutual Information\n")
    f.write(str(normalized_mutual_info_score(assign_topics, labels)))
    f.write("\nTopic Purity\n")
    f.write(str(purity_score(assign_topics, labels)))
    f.write("\nTopic Coherence\n")
    f.write(str(topic_coherence(topics, texts, voca, 15)))
    f.write("\nTopic SVM Accuracy\n")
    scores = cross_val_score(clf, dist_topics, labels, cv=k - 1)
    f.write("%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    f.write("\nTopic SVM F1_micro\n")
    scores = cross_val_score(clf, dist_topics, labels,
                             cv=k - 1, scoring='f1_micro')
    f.write("%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    f.write("\nTopic Precison\n")
    f.write(str(precision_score(labels, assign_topic_label(
        assign_topics, labels), average='micro')))
    f.write("\nTopic Recall\n")
    f.write(str(recall_score(labels, assign_topic_label(
        assign_topics, labels), average='macro')))
    f.close
