import configparser
from text_processor import TextProcessor
import topic_BTM as btm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import manifold
from matplotlib import pyplot as plt
import numpy as np
import math
import csv


def coherence_value(words, docs, vocab):
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

if __name__ == '__main__':
cf = configparser.ConfigParser()
cf.read("file_path.properties")
path = dict(cf.items("file_path"))
dir_btm = path['dir_btm']
dir_in = path['dir_in']
dir_pln = path['dir_pln']
tp = TextProcessor()

voca = btm.read_voca(dir_btm + "voca.txt")
inv_voca = {v: k for k, v in voca.items()}
pz_pt = dir_btm + "/model/k3.pz"
pz = btm.read_pz(pz_pt)
zw_pt = dir_btm + '/model/k3.pw_z'

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

texts = list()
tweets = open(dir_in + "tweets.txt", "r")
for l in tweets:
    texts.append(l.split())


dist_topics = list()
assing_topics = list()
tw_l = list()
for t in texts:
    tw_topics = list()
    lista = list()
    for tp in topics:
        tw = dict()
        tp = dict(tp)
        tmp = 0
        for w in t:
            tw[w] = tp[inv_voca[w]]
            tmp += tp[inv_voca[w]]
        tw_topics.append(tmp)
        lista.append(tw)
    tw_l.append(lista)
    assing_topics.append(tw_topics.index(max(tw_topics)))
    dist_topics.append([round(x / sum(tw_topics), 2) for x in tw_topics])

wntm = list()
tw = open(dir_in + "tweets.txt", "r")
for l in tw:
    wntm.append(l.split())

topic_coherence(topics, wntm, voca, 10)
topic_coherence(topics, texts, voca, 10)


x = [2, 1,0 ]
r = np.repeat(x, 100)
recall_score(r, assing_topics, average='macro')
precision_score(r, assing_topics, average='macro')

[voca[x] for (x, v) in topics[0][:10]]
[voca[x] for (x, v) in topics[1][:10]]
[voca[x] for (x, v) in topics[2][:10]]

errou = list()
for i, v in enumerate(assing_topics):
        errou.append(tw_l[i][v])

certo = list()
for i, v in enumerate(r):
        certo.append(tw_l[i][v])


l = list()
for i, v in enumerate(assing_topics):
    if(v != r[i]):
        l.append(i)

m = [dist_topics[x] for x in l]
k = [r[x] for x in l]
p = [assing_topics[x] for x in l]
j = [" ".join(texts[x]) for x in l]
h = [errou[x] for x in l]
f = [certo[x] for x in l]

np.column_stack((k, p, m, h,f))

tw = open(dir_in + "tweets.txt", "r")
lda_top =list()
for l in tw:
    n =[i for x,i in ldamodel.get_document_topics(dic.doc2bow(l.split()))]
    lda_top.append(n.index(max(n)))

txt = list()
tw = open(dir_in + "tweets.adjacent", "r")
for l in tw:
    txt.append(l.split())


x = [2, 0,1 ]
r = np.repeat(x, 100)
recall_score(r, lda_top, average='macro')
precision_score(r, lda_top, average='macro')


with open(dir_pln+'dic_tfidf_like.pck','rb') as handle:
    tfidf = dict(pickle.load(handle))


# f = open(dir_pln+'tfidf_like_parametros.csv', 'rt')
# reader = csv.reader(f, delimiter=';')

# tfidf = dict()
# for row in reader:
#     tfidf[row[0]] = double(row[1])

topics2= list()
for tp in topics:
    tmp = dict()
    for tw in tp:
        tmp[tw[0]] = (tw[1] * tfidf[voca(tw[0])])
    topics2.append(tmp)
        