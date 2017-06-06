#!/usr/bin/env python
# coding=utf-8
# Function: translate the results from BTM
# Input:
#    mat/pw_z.k20

import sys
import networkx as nx


# return:    {wid:w, ...}
def read_voca(pt):
    voca = {}
    for l in open(pt):
        wid, w = l.strip().split('\t')[:2]
        voca[int(wid)] = w
    return voca


def read_pz(pt):
    return [float(p) for p in open(pt).readline().split()]
# voca = {wid:w,...}


def dispTopics(pt, voca, pz):
    k = 0
    topics = []
    tp_graph = []
    for l in open(pt):
        vs = [float(v) for v in l.split()]
        wvs = zip(range(len(vs)), vs)
        wvs = sorted(wvs, key=lambda d: d[1], reverse=True)
        # tmps = ' '.join(['%s' % voca[w] for w,v in wvs[:10]])
        tp_graph.append([w for w, v in wvs[:15]])
        tmps = ' '.join(['%s:%f' % (voca[w], v) for w, v in wvs[:10]])
        topics.append((pz[k], tmps))
        k += 1

    print('p(z)\t\tTop words')
    for pz, s in sorted(topics, reverse=True):
        print('%f\t%s' % (pz, s))

    print('building graph')
    # graph(tp_graph)


def graph(topics):
    topic_graph = nx.Graph()
    for i in range(len(topics)):
        for k in range(i + 1, len(topics)):
            weight = round(len(set(topics[i]) & set(
                topics[k])) / len(topics[0]), 3)
            topic_graph.add_edge(i, k, weight=weight)
            nx.write_gml(topic_graph, "topicos.gml")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python %s <model_dir> <K> <voca_pt>' % sys.argv[0])
        print('\tmodel_dir    the output dir of BTM')
        print('\tK    the number of topics')
        print('\tvoca_pt    the vocabulary file')
        exit(1)

    model_dir = sys.argv[1]
    K = int(sys.argv[2])
    voca_pt = sys.argv[3]
    voca = read_voca(voca_pt)
    W = len(voca)
    print('K:%d, n(W):%d' % (K, W))

    pz_pt = model_dir + 'k%d.pz' % K
    pz = read_pz(pz_pt)

    zw_pt = model_dir + 'k%d.pw_z' % K
    dispTopics(zw_pt, voca, pz)



