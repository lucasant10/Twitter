import sys
sys.path.append('../')
import configparser
import topic_BTM as btm
import numpy as np
from collections import defaultdict

if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_btm = path['dir_btm']
    dir_in = path['dir_in']

    print("Reading vocab ")
    p_voca = btm.read_voca(dir_btm + "voca.txt")
    p_inv_voca = {v: k for k, v in p_voca.items()}

    n_p_voca = btm.read_voca(dir_btm + "voca2.txt")
    n_p_inv_voca = {v: k for k, v in n_p_voca.items()}

    print("Loading politcs_text ")
    politcs_text = list()
    tweets = open(dir_in + "politics_text.txt", "r")
    for l in tweets:
        politcs_text.append(l.split())

    print("Loading non politcs_text ")
    n_politcs_text = list()
    tweets = open(dir_in + "non_politics_text.txt", "r")
    for l in tweets:
        n_politcs_text.append(l.split())

    dist_politics = list()
    dist_n_politics = list()

    assing_topics = list()
    print("Reading topic")
    pz_pt = dir_btm + "model/k10.pz"
    pz = btm.read_pz(pz_pt)
    zw_pt = dir_btm + "model/k10.pw_z"

    print("Processing topic")
    k = 0
    topics = []
    vectors = np.zeros((len(pz), len(p_inv_voca)))
    for k, l in enumerate(open(zw_pt)):
        vs = [float(v) for v in l.split()]
        vectors[k, :] = vs
        wvs = zip(range(len(vs)), vs)
        wvs = dict(sorted(wvs, key=lambda d: d[1], reverse=True))
        topics.append(wvs)
        k += 1

    print("Assing politics topics")
    assing_politcs = defaultdict(int)
    for t in politcs_text:
        tw_topics = list()
        for tp in topics:
            tmp = 0
            for w in t:
                if w in p_inv_voca:
                    tmp += tp[p_inv_voca[w]]
            tw_topics.append(tmp)

        assing_politcs[tw_topics.index(max(tw_topics))] += 1

    print("Reading topic")
    pz_pt = dir_btm + "model2/k10.pz"
    pz = btm.read_pz(pz_pt)
    zw_pt = dir_btm + "model2/k10.pw_z"

    print("Processing topic")
    k = 0
    topics = []
    vectors = np.zeros((len(pz), len(n_p_inv_voca)))
    for k, l in enumerate(open(zw_pt)):
        vs = [float(v) for v in l.split()]
        vectors[k, :] = vs
        wvs = zip(range(len(vs)), vs)
        wvs = dict(sorted(wvs, key=lambda d: d[1], reverse=True))
        topics.append(wvs)
        k += 1 

    print("Assing non politics topics")
    assing_n_politcs = defaultdict(int)
    for t in n_politcs_text:
        tw_topics = list()
        for tp in topics:
            tmp = 0
            for w in t:
                if w in n_p_inv_voca:
                    tmp += tp[n_p_inv_voca[w]]
            tw_topics.append(tmp)
        assing_n_politcs[tw_topics.index(max(tw_topics))] += 1

    print("Saving file")
    total = sum(assing_politcs.values()) + sum(assing_n_politcs.values())
    f = open(dir_in + "btm_politicos.txt", 'w')
    f.write("-- political topics -- \n\n")
    for i in range(0, len(topics)):
        f.write("%0.2f\n" % (assing_politcs[i] / total * 100))

    f.write("\n\n-- non political topics -- \n\n")
    for i in range(0, len(topics)):
        f.write("%0.2f\n" % (assing_n_politcs[i] / total * 100))

    f.close()
