import sys
sys.path.append('../')
import configparser
import topic_BTM as btm
import numpy as np
from matplotlib import pyplot as plt


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

def kl(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0,(p-q) * np.log10(p / q), 0))


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_btm = path['dir_btm']
    dir_in = path['dir_in']


    print("Reading vocab " )
    voca = btm.read_voca(dir_btm + "voca.txt")
    inv_voca = {v: k for k, v in voca.items()}

    print("Loading baseline file " )
    baseline = list()
    tweets = open(dir_in + "tweets_nao_politicos.txt", "r")
    for l in tweets:
        baseline.append(l.split())

    dist_topics = list()
    k_list = [2,3,4,5]
    for i in k_list:
        assing_topics = list()
        print("Reading topic %s" %i )
        pz_pt = dir_btm + "output\k"+ str(i) +".pz"
        pz = btm.read_pz(pz_pt)
        zw_pt = dir_btm + "output\k"+ str(i) +".pw_z"

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

    max_coef = list()
    gini_coef = list()
    f =  open(dir_in+"btm_nao_politicos.txt", 'w')
    for i, dist in enumerate(dist_topics):
        counter = np.bincount(dist)
        dist = np.round( counter/np.sum(counter), 2)
        max_coef.append(max(dist))
        gini_coef.append( gini(dist))
        f.write("topic %s: " %(i) + str(dist) + "\n\n")
    f.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)    
    ax.set_title("Tweets Politicos")
    ax.set_xlabel("Topicos K")
    ax.set_ylabel("Max da distribuicao")
    ax.set_ylim(0,1)
    ax.plot(k_list,max_coef,'-o')
    plt.show()
    plt.clf()


    fig = plt.figure()
    ax = fig.add_subplot(111)    
    ax.set_title("Tweets Politicos")
    ax.set_xlabel("Topicos K")
    ax.set_ylabel("Gini coefficient")
    ax.set_ylim(0,1)
    ax.plot(k_list,gini_coef,'-o')
    plt.show()
    plt.clf()
