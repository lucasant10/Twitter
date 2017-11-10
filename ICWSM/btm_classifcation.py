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

    print("Loading politcs_text " )
    politcs_text = list()
    tweets = open(dir_in + "tweets_politicos.txt", "r")
    for l in tweets:
        politcs_text.append(l.split())

    print("Loading non politcs_text " )
    n_politcs_text = list()
    tweets = open(dir_in + "tweets_nao_politicos.txt", "r")
    for l in tweets:
        n_politcs_text.append(l.split())

    dist_politics = list()
    dist_n_politics = list()
    k_list = [2,3,4,5]
    for i in k_list:
        assing_topics = list()
        print("Reading topic %s" %i )
        pz_pt = dir_btm + "model/k"+ str(i) +".pz"
        pz = btm.read_pz(pz_pt)
        zw_pt = dir_btm + "model/k"+ str(i) +".pw_z"

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
            assing_politcs.append(tw_topics.index(max(tw_topics)))
        dist_politics.append(assing_politcs)


        print("Assing non politics topics")
        assing_n_politcs = list()
        for t in n_politcs_text:
            tw_topics = list()
            for tp in topics:
                tmp = 0
                for w in t:
                    if w in inv_voca:
                        tmp += tp[inv_voca[w]]
                tw_topics.append(tmp)
            assing_n_politcs.append(tw_topics.index(max(tw_topics)))
        dist_n_politics.append(assing_n_politcs)


    print("Saving files")
    max_coef_politics = list()
    gini_coef_politics = list()
    dist_kl_politics = list()
    idx_politics = list()
    f = open(dir_in+"btm_politicos.txt", 'w')
    f.write("-- political topics distribution -- \n\n")
    for i, dist in enumerate(dist_politics):
        c_politics = np.bincount(dist)
        politics_dist = np.round(c_politics/np.sum(c_politics), 2)
        z = list(zip(range(len(politics_dist)), politics_dist))
        print(politics_dist)
        dist_kl_politics.append(z)
        max_coef_politics.append(max(politics_dist))
        idx_politics.append(np.argmax(politics_dist))
        gini_coef_politics.append(gini(politics_dist))
        f.write("topic %s: " % (i+2) + str(['%0.2f' %x[1] for x in z]) + "\n\n")

    f.write("-- non political topics distribution -- \n\n")
    max_coef_n_politics = list()
    gini_coef_n_politics = list()
    dist_kl_n_politics = list()
    idx_n_politics = list()
    for i, dist in enumerate(dist_n_politics):
        c_n_politics = np.bincount(dist)
        politics_n_dist = np.round(c_n_politics/np.sum(c_n_politics), 2)
        z = list(zip(range(len(politics_n_dist)), politics_n_dist))
        print(politics_n_dist)
        dist_kl_n_politics.append(z)
        max_coef_n_politics.append(max(politics_n_dist))
        idx_n_politics.append(np.argmax(politics_n_dist))
        gini_coef_n_politics.append(gini(politics_n_dist))
        f.write("topic %s: " % (i+2) +  str(['%0.2f' %x[1] for x in z])  + "\n\n")
   
    f.write("-- KL Divergence -- \n\n")
    for i, dist in enumerate(dist_kl_politics):
        p = [x[1] for x in dist]
        q = [y[1] for y in dist_kl_n_politics[i]]
        f.write("Kl Topicos %s: %s \n" %((i+2), np.round(kl(p,q), 2)))

    f.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Tweets Politicos")
    ax.set_xlabel("Topicos K")
    ax.set_ylabel("Max da distribuicao")
    ax.set_ylim(0, 1)
    ax.plot(k_list, max_coef_politics, '-o', color="red", label='politicos')
    ax.plot(k_list, max_coef_n_politics, '-o', color ="blue", label='nao politicos')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    for i, idx in enumerate(idx_politics):
        ax.annotate('%s' %idx, xy=(k_list[i],max_coef_politics[i]), xytext=(0,-10), textcoords='offset points')
    for i, idx in enumerate(idx_n_politics):
        ax.annotate('%s' %idx, xy=(k_list[i],max_coef_n_politics[i]), xytext=(0,5), textcoords='offset points')
    plt.show()
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Tweets Politicos")
    ax.set_xlabel("Topicos K")
    ax.set_ylabel("Gini coefficient")
    ax.set_ylim(0, 1)
    ax.plot(k_list, gini_coef_politics, '-o', color="red", label='politicos')
    ax.plot(k_list, gini_coef_n_politics, '-o', color ="blue",  label='nao politicos')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    for i, idx in enumerate(idx_politics):
        ax.annotate('%s' %idx, xy=(k_list[i],gini_coef_politics[i]), xytext=(0,-10), textcoords='offset points')
    for i, idx in enumerate(idx_n_politics):
        ax.annotate('%s' %idx, xy=(k_list[i],gini_coef_n_politics[i]), xytext=(0,5), textcoords='offset points')
    plt.show()
    plt.clf()

