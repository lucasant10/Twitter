import sys
sys.path.append('../')
import configparser
import topic_BTM as btm
import numpy as np
from collections import defaultdict


class AssingTopics():

    def __init__(self, dir_btm, dir_in, voca, pz_pt_path, zw_pt_path):
        self.dir_btm = dir_btm
        self.dir_in = dir_in
        self.voca, self.inv_voca = self.vocab(voca)
        self.pz_pt = dir_btm + pz_pt_path
        self.pz = btm.read_pz(self.pz_pt)
        self.zw_pt = dir_btm + zw_pt_path
        self.topics = self.get_topics(
            self.pz, self.inv_voca, self.voca, self.zw_pt)

    def vocab(self, path):
        voca = btm.read_voca(self.dir_btm + path)
        inv_voca = {v: k for k, v in voca.items()}
        return voca, inv_voca

    def get_topics(self, pz, inv_voca, voca, zw_pt):
        topics = list()
        vectors = np.zeros((len(pz), len(inv_voca)))
        for k, l in enumerate(open(zw_pt)):
            vs = [float(v) for v in l.split()]
            vectors[k, :] = vs
            wvs = zip(range(len(vs)), vs)
            wvs = dict(sorted(wvs, key=lambda d: d[1], reverse=True))
            topics.append(wvs)
        return topics

    def load_file(self, dir_in, file_name):
        text = list()
        tweets = open(dir_in + file_name, "r")
        for l in tweets:
            text.append(l.split())
        return text

    def get_topic_distribution(self, text, topics):
        assing = defaultdict(int)
        for t in text:
            tw_topics = list()
            for tp in topics:
                tmp = 0
                for w in t:
                    if w in self.inv_voca:
                        tmp += tp[self.inv_voca[w]]
                tw_topics.append(tmp)
            assing[tw_topics.index(max(tw_topics))] += 1
        return assing

    def adjetive_index(self, text, topics, topic_list):
        total = defaultdict(int)
        target = defaultdict(int)
        for t in text:
            tw_topics = list()
            for tp in topics:
                tmp = 0
                for w in t:
                    if w in self.inv_voca:
                        tmp += tp[self.inv_voca[w]]
                tw_topics.append(tmp)
            index = tw_topics.index(max(tw_topics))
            if index in topic_list:
                target[index] += 1
            total[index] += 1
        return (total, target)

    def get_text_topic(self, text, topics):
        tw_topics = list()
        for tp in topics:
            tmp = 0
            for w in text:
                if w in self.inv_voca:
                    tmp += tp[self.inv_voca[w]]
            tw_topics.append(tmp)
        return tw_topics.index(max(tw_topics))