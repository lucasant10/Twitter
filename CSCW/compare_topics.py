import sys
sys.path.append('../')
import configparser
import topic_BTM as btm
import numpy as np
from collections import defaultdict
import networkx as nx
from assign_topics import AssingTopics


def vocab(path):
    voca = btm.read_voca(dir_btm + path)
    inv_voca = {v: k for k, v in voca.items()}
    return voca, inv_voca


def processing_topic(pz, inv_voca, voca, zw_pt, top_k):
    topics = []
    vectors = np.zeros((len(pz), len(inv_voca)))
    for k, l in enumerate(open(zw_pt)):
        vs = [float(v) for v in l.split()]
        vectors[k, :] = vs
        wvs = zip(range(len(vs)), vs)
        wvs = sorted(wvs, key=lambda d: d[1], reverse=True)
        topics.append([voca[w] for w, v in wvs[:top_k]])
        k += 1
    return topics


def jaccard_distance(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)


def jaccard_topics(t_base, topics):
    tmp = list()
    for topic in topics:
        tmp.append(jaccard_distance(t_base, topic))
    return (tmp.index(max(tmp)), max(tmp))


def list2text(topics):
    txt = ''
    for k, topic in enumerate(topics):
        txt += "%d :" % (k + 1) + ' '.join(topic) + '\n'
    return txt


def load_file(dir_in, file_name):
    text = list()
    tweets = open(dir_in + file_name, "r")
    for l in tweets:
        text.append(l.split())
    return text


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_btm = path['dir_btm']
    dir_in = path['dir_in']

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
    p_topics = processing_topic(p_pz, p_inv_voca, p_voca, p_zw_pt, 10)
    np_topics = processing_topic(np_pz, n_p_inv_voca, n_p_voca, np_zw_pt, 10)
    a_topics = processing_topic(a_pz, all_inv_voca, all_voca, a_zw_pt, 10)

    print("processing assign topic distribution")
    at = AssingTopics(dir_btm, dir_in, 'voca.txt',
                      "model/k10.pz", "model/k10.pw_z")
    dist_pol = at.get_topic_distribution(
        load_file(dir_in, 'politics.txt'), at.topics)
    at = AssingTopics(dir_btm, dir_in, 'voca2.txt',
                      "model2/k10.pz", "model2/k10.pw_z")
    dist_n_pol = at.get_topic_distribution(
        load_file(dir_in, 'non_politics.txt'), at.topics)
    at = AssingTopics(dir_btm, dir_in, 'all_voca2.txt',
                      "model2/k20.pz", "model2/k20.pw_z")
    dist_both = at.get_topic_distribution(
        load_file(dir_in, 'both_politics.txt'), at.topics)

    joint_topics = p_topics + np_topics
    total = sum(dist_both.values())
    p_total = sum(dist_pol.values())
    np_total = sum(dist_n_pol.values())
    print("creating graph")
    t_graph = nx.Graph()
    t_graph.add_nodes_from([('all %d' % (k + 1), dict(perc='%0.2f' % (dist_both[k] / total)))
                            for k, v in enumerate(a_topics)], bipartite=0)
    t_graph.add_nodes_from([('pol %d' % (k + 1), dict(perc='%0.2f' % (dist_pol[k] / total)))
                            for k, v in enumerate(p_topics)], bipartite=1)
    t_graph.add_nodes_from([('n_pol %d' % (k + 1), dict(perc='%0.2f' % (dist_n_pol[k] / total)))
                            for k, v in enumerate(np_topics)], bipartite=1)
    txt = ''
    for k, tp in enumerate(a_topics):
        index, value = jaccard_topics(tp, joint_topics)
        if index < 10:
            t_graph.add_edge('all %d' % (k + 1), 'pol %d' %
                             (index + 1), weight=value)
            txt += 'all %d -> pol %d : W = %0.2f, %%(%0.2f, %0.2f)\n' % (
                (k + 1), (index + 1), value, (dist_both[k] / total), (dist_pol[index] / total))
        else:
            t_graph.add_edge('all %d' % (k + 1), 'n_pol %d' %
                             (index - 9), weight=value)
            txt += 'all %d -> n_pol %d : W = %0.2f, %%(%0.2f, %0.2f)\n' % (
                (k + 1), (index - 9), value, (dist_both[k] / total), (dist_n_pol[(index - 10)] / total))

    print("Saving graph file")
    nx.write_gml(t_graph, dir_in + "topics_graph.gml")
    f = open(dir_in + "compare_topics.txt", 'w')
    f.write(txt)
    f.write("\n\n-- political topics -- \n\n")
    f.write(list2text(p_topics))
    f.write("\n\n-- non_political topics -- \n\n")
    f.write(list2text(np_topics))
    f.write("\n\n-- both topics -- \n\n")
    f.write(list2text(a_topics))
    f.close()

    # # Separate by group
    # l, r = nx.bipartite.sets(t_graph)
    # pos = {}

    # # Update position for node from each group
    # pos.update((node, (1, index)) for index, node in enumerate(l))
    # pos.update((node, (2, index)) for index, node in enumerate(r))

    # nx.draw(B, pos=pos)
    # plt.show()
