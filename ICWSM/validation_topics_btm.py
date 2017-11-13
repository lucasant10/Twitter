import sys
sys.path.append('../')
import configparser
import gensim
import numpy as np
from sklearn.metrics import classification_report
import topic_BTM as btm


if __name__ == '__main__':

    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_btm = path['dir_btm']
    dir_in = path['dir_in']

    print("Reading vocab " )
    voca = btm.read_voca(dir_btm + "voca_wntm.txt")
    inv_voca = {v: k for k, v in voca.items()}

    print("Loading politcs_text " )
    politcs_text = list()
    tweets = open(dir_in + "politicos.txt", "r")
    for l in tweets:
        politcs_text.append(l.split())

    print("Loading non politcs_text " )
    n_politcs_text = list()
    tweets = open(dir_in + "nao_politicos.txt", "r")
    for l in tweets:
        n_politcs_text.append(l.split())

    k=2
    print("Assing politics topics")
    y_pred = list()
    y_true = list()
    print("Reading topic %s" %k )
    pz_pt = dir_btm + "model_wntm/k"+ str(k) +".pz"
    pz = btm.read_pz(pz_pt)
    zw_pt = dir_btm + "model_wntm/k"+ str(k) +".pw_z"

    print("Processing topic %s" %k )
    topics = []
    vectors = np.zeros((len(pz), len(inv_voca)))
    for i, l in enumerate(open(zw_pt)):
        vs = [float(v) for v in l.split()]
        vectors[i, :] = vs
        wvs = zip(range(len(vs)), vs)
        wvs = dict(sorted(wvs, key=lambda d: d[1], reverse=True))
        topics.append(wvs)


    print("Assing politics topics")
    assing_politcs = list()
    for t in politcs_text:
        tw_topics = list()
        for tp in topics:
            tmp = 0
            for w in t:
                if w in inv_voca:
                    tmp += tp[inv_voca[w]]
            tw_topics.append(tmp)
        y_pred.append(tw_topics.index(max(tw_topics)))
        y_true.append(0)


    assing_n_politcs = list()
    for t in politcs_text:
        tw_topics = list()
        for tp in topics:
            tmp = 0
            for w in t:
                if w in inv_voca:
                    tmp += tp[inv_voca[w]]
            tw_topics.append(tmp)
        y_pred.append(tw_topics.index(max(tw_topics)))
        y_true.append(1)

    print(classification_report(y_true, y_pred))
   