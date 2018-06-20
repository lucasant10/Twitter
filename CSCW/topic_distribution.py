import sys
sys.path.append('../')
import configparser
import topic_BTM as btm
import numpy as np
from collections import Counter, defaultdict
from assign_topics import AssingTopics
import pymongo
from political_classification import PoliticalClassification
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def vocab(path):
    voca = btm.read_voca(dir_btm + path)
    inv_voca = {v: k for k, v in voca.items()}
    return voca, inv_voca


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_btm = path['dir_btm']
    dir_in = path['dir_in']

    person_topicos = [0,1,2,3,4,5,6,7,8,9]

    print("Reading vocab ")
    all_voca, all_inv_voca = vocab('voca2.txt')

    assing_topics = list()
    print("Reading topic")
    a_pz_pt = dir_btm + "model2/k10.pz"
    a_pz = btm.read_pz(a_pz_pt)
    a_zw_pt = dir_btm + "model2/k10.pw_z"

    print("processing assign topic distribution")
    at = AssingTopics(dir_btm, dir_in, 'voca2.txt',
                      "model2/k10.pz", "model2/k10.pw_z")

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    tweets = db.tweets.find({'created_at': {
                            '$gte': 1380596400000, '$lt': 1443668400000}, 'cond_55': {'$exists': True}})

    pc = PoliticalClassification('cnn_s300.h5', 'cnn_s300.npy', 18)

    chosen = list()
    total = 0
    for tweet in tweets:
        total += 1
        if not pc.is_political(tweet['text_processed']):
            chosen.append(tweet['text_processed'].split())

    topic_dist = at.get_topic_distribution(chosen, at.topics)

    txt = ''
    sorted_dict = sorted(topic_dist)
    for idx in sorted_dict:
        z = (topic_dist[idx] / total) if topic_dist[idx] != 0 else 0
        txt += 'idx %d: %0.2f%%\n' % ((idx+1),(z * 100))

    print("Saving file")
    f = open(dir_in + "np_topics.txt", 'w')
    f.write(txt)
    f.close()

