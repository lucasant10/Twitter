import sys
sys.path.append('../')
import configparser
import gensim
import numpy as np
from matplotlib import pyplot as plt


def gini(data):
    def _unit_area(height, value, width):
        bar_area = (height * width) + ((value * width) / 2.)
        return bar_area
    fair_area = 0.5
    datasum = float(sum(data))
    if datasum == 0:
        import sys
        m = 'Data sum is 0.0.\nCannot calculate Gini coefficient for non-responsive population.'
        print(m)
        sys.exit()
    if datasum != 1.0:
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
    return np.sum(np.where(p != 0, (p-q) * np.log10(p / q), 0))

if __name__ == '__main__':

    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_lda = path['dir_lda']
    dir_in = path['dir_in']

    dist_n_politics = list()
    dist_politics = list()
    k_list = [2, 3, 4]

    for k in k_list:
        print("processing model %sk " % k)
        model = gensim.models.LdaModel.load(dir_lda + "wntm_model_%sk.lda" % k)
        vocab = model.id2word

        print("Loading tweets_politicos file ")
        bow_politics = list()
        tweets = open(dir_in + "tweets_politicos.txt", "r")
        for l in tweets:
            bow_politics.append(vocab.doc2bow(l.split()))

        print("Loading tweets_nao_politicos file ")
        bow_n_politics = list()
        tweets = open(dir_in + "tweets_nao_politicos.txt", "r")
        for l in tweets:
            bow_n_politics.append(vocab.doc2bow(l.split()))

        print("Assing politics topics")
        assing_politcs = list()
        assing_n_politcs = list()
        for txt in bow_politics:
            assing_politcs.append(np.argmax([x[1] for x in model[txt]]))
        dist_politics.append(assing_politcs)

        print("Assing not politics topics")
        assing_n_politcs = list()
        for txt in bow_n_politics:
            assing_n_politcs.append(np.argmax([x[1] for x in model[txt]]))
        dist_n_politics.append(assing_n_politcs)

    print("Saving files")
    max_coef_politics = list()
    gini_coef_politics = list()
    dist_kl_politics = list()
    f = open(dir_in+"wntm_politicos.txt", 'w')
    f.write("-- political topics distribution -- \n\n")
    for i, dist in enumerate(dist_politics):
        c_politics = np.bincount(dist)
        politics_dist = np.round(c_politics/np.sum(c_politics), 2)
        dist_kl_politics.append(politics_dist)
        max_coef_politics.append(max(politics_dist))
        gini_coef_politics.append(gini(politics_dist))
        f.write("topic %s: " % (i+2) + str(politics_dist) + "\n\n")

    f.write("-- non political topics distribution -- \n\n")
    max_coef_n_politics = list()
    gini_coef_n_politics = list()
    dist_kl_n_politics = list()
    for i, dist in enumerate(dist_n_politics):
        c_n_politics = np.bincount(dist)
        politics_n_dist = np.round(c_n_politics/np.sum(c_n_politics), 2)
        dist_kl_n_politics.append(politics_n_dist)
        max_coef_n_politics.append(max(politics_n_dist))
        gini_coef_n_politics.append(gini(politics_n_dist))
        f.write("topic %s: " % (i+2) + str(politics_n_dist) + "\n\n")
   
    f.write("-- KL Divergence -- \n\n")

    for i, dist in enumerate(dist_kl_politics):
        f.write("Kl Topicos %s: %s \n" %((i+2), np.round(kl(dist, dist_kl_n_politics[i]), 2)))

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
    plt.show()
    plt.clf()
