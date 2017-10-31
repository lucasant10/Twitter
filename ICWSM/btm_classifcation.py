import sys
sys.path.append('../')
import configparser
from text_processor import TextProcessor
import topic_BTM as btm
import numpy as np

if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_btm = path['dir_btm']
    dir_in = path['dir_in']
    dir_out = path['dir_out']

    tp = TextProcessor()

    print("Reading vocab " )
    voca = btm.read_voca(dir_btm + "voca.txt")
    inv_voca = {v: k for k, v in voca.items()}

    print("Loading baseline file " )
    baseline = list()
    tweets = open(dir_out + "tweets_baseline.txt", "r")
    for l in tweets:
        baseline.append(l.split())

    dist_topics = list()
    for i in range(10, 40, 10):
        assing_topics = list()
        print("Reading topic %s" %i )
        pz_pt = dir_btm + "/model/k"+ str(i) +".pz"
        pz = btm.read_pz(pz_pt)
        zw_pt = dir_btm + "/model/k"+ str(i) +".pw_z"

        print("Processing topic %s" %i )
        k = 0
        topics = []
        vectors = np.zeros((len(pz), len(inv_voca)))
        for k, l in enumerate(open(zw_pt)):
            vs = [float(v) for v in l.split()]
            vectors[k, :] = vs
            wvs = zip(range(len(vs)), vs)
            wvs = dict(sorted(wvs, key=lambda d: d[1], reverse=True))
            topics.append(wvs)
            k += 1

        print("Reading baseline file")
        for t in baseline:
            tw_topics = list()
            for tp in topics:
                tmp = 0
                for w in t:
                    if w in inv_voca:
                        tmp += tp[inv_voca[w]]
                tw_topics.append(tmp)
            assing_topics.append(tw_topics.index(max(tw_topics)))
        dist_topics.append(assing_topics)

    print("Saving file btm_dist_topics")

    f =  open(dir_out+"coleta/btm_dist_topics.txt", 'w')
    for i, dist in enumerate(dist_topics):
        counter = np.bincount(dist)
        f.write("topic %s: " %((i+1)*10) + str(np.round( counter/np.sum(counter), 2) ) + "\n\n")
    f.close()