import sys
sys.path.append('../')
import configparser
import topic_BTM as btm
import numpy as np
from collections import defaultdict
import networkx as nx


def vocab(path):
    voca = btm.read_voca(dir_btm + path)
    inv_voca = {v: k for k, v in voca.items()}
    return voca, inv_voca


def processing_topic(pz, inv_voca, zw_pt):
    topics = []
    vectors = np.zeros((len(pz), len(inv_voca)))
    for k, l in enumerate(open(zw_pt)):
        vs = [float(v) for v in l.split()]
        vectors[k, :] = vs
        wvs = zip(range(len(vs)), vs)
        wvs = dict(sorted(wvs, key=lambda d: d[1], reverse=True))
        topics.append(wvs)
        k += 1
    return topics


def topic_index(tweet, topics, inv_voca):
    for tp in topics:
        tmp = 0
        for w in tweet:
            if w in inv_voca:
                tmp += tp[inv_voca[w]]
        tw_topics.append(tmp)
    return tw_topics.index(max(tw_topics))


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_btm = path['dir_btm']
    dir_in = path['dir_in']

    pc = PoliticalClassification('cnn_s300.h5', 'cnn_s300.npy', 18)

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    tweets = db.tweets.find({'created_at': {
                            '$gte': 1380585600000, '$lt': 1443830400000}, 'cond_55': {'$exists': True}})

    voca = ['voca.txt', 'voca2.txt', 'all_voca2.txt']

    print("Reading vocab ")
    p_voca, p_inv_voca = vocab('voca.txt')
    n_p_voca, n_p_inv_voca = vocab('voca2.txt')
    all_voca, all_inv_voca = vocab('all_voca2.txt')

    assing_topics = list()
    print("Reading topic")
    p_pz_pt = dir_btm + "model/k10.pz"
    p_pz = btm.read_pz(p_pz_pt)
    p_zw_pt = dir_btm + "model/k10.pw_z"

    np_pz_pt = dir_btm + "model2/k10.pz"
    np_pz = btm.read_pz(np_pz_pt)
    np_zw_pt = dir_btm + "model2/k10.pw_z"

    a_pz_pt = dir_btm + "model2/k20.pz"
    a_pz = btm.read_pz(a_pz_pt)
    a_zw_pt = dir_btm + "model2/k20.pw_z"

    print("Processing topic")
    p_topics = processing_topic(p_pz, p_inv_voca, p_zw_pt)
    np_topics = processing_topic(np_pz, np_inv_voca, np_zw_pt)
    a_topics = processing_topic(a_pz, a_inv_voca, a_zw_pt)

    print("creating graph")
    t_graph = nx.Graph()
    for tweet in tweets:
        all_index = topic_index(tweet, a_topics, a_inv_voca)
        if pc.is_political(tweet):
            p_index = topic_index(tweet, p_topics, p_inv_voca)
            t_graph.add_edge('all %d' % all_index, 'pol %d' % p_index)
        else:
            np_index = topic_index(tweet, np_topics, np_inv_voca)
            t_graph.add_edge('all %d' % all_index, 'non_pol %d' % np_index)

    print("Saving graph file")
    nx.write_gml(t_graph, dir_in + "topics_graph.gml")
