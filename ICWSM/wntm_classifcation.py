import sys
sys.path.append('../')
import configparser
from text_processor import TextProcessor
import topic_BTM as btm
import numpy as np


def gini(data):
    def _unit_area(height, value, width):
        bar_area = (height * width) + ((value * width) / 2.) 
        return bar_area      
    fair_area = 0.5 
    datasum = float(sum(data))
    if datasum==0:
        import sys
        m = 'Data sum is 0.0.\nCannot calculate Gini coefficient for non-responsive population.' 
        print(m)
        sys.exit()
    if datasum!=1.0:
        data = [x/datasum for x in data]
    data.sort()
    width = 1/float(len(data))
    height, area = 0.0, 0.0 
    for value in data:
        area += _unit_area(height, value, width)
        height += value
    gini = (fair_area-area)/fair_area
    return gini

if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_btm = path['dir_btm']
    dir_in = path['dir_in']
    dir_out = path['dir_out']

    tp = TextProcessor()

    print("Reading vocab " )
    voca = btm.read_voca(dir_btm + "voca_wntm.txt")
    inv_voca = {v: k for k, v in voca.items()}

    print("Loading baseline file " )
    baseline = list()
    tweets = open(dir_out + "tweets_baseline.txt", "r")
    for l in tweets:
        baseline.append(l.split())

    dist_topics = list()
    for i in range(2, 25, 5):
        assing_topics = list()
        print("Reading topic %s" %i )
        pz_pt = dir_btm + "/model_wntm/k"+ str(i) +".pz"
        pz = btm.read_pz(pz_pt)
        zw_pt = dir_btm + "/model_wntm/k"+ str(i) +".pw_z"

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

        print("Assing topics to baseline tweets")
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

    k_list = list()
    max_coef = list()
    gini_coef = list()
    f =  open(dir_out+"coleta/wntm_dist_topics.txt", 'w')
    for i, dist in enumerate(dist_topics):
        topic = (i + 1) * 5
        counter = np.bincount(dist)
        dist = np.round( counter/np.sum(counter), 2)
        max_coef = max(dist)
        gini_coef = gini(dist)
        k_list = topic
        f.write("topic %s: " %(topic) + str(dist) + "\n\n")
    f.close()
