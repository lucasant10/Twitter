import sys
sys.path.append('../')
import configparser
import topic_BTM as btm
import numpy as np
from collections import defaultdict
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
    all_voca, all_inv_voca = vocab('all_voca2.txt')

    assing_topics = list()
    print("Reading topic")
    a_pz_pt = dir_btm + "model2/k20.pz"
    a_pz = btm.read_pz(a_pz_pt)
    a_zw_pt = dir_btm + "model2/k20.pw_z"

    print("processing assign topic distribution")
    at = AssingTopics(dir_btm, dir_in, 'all_voca2.txt',
                      "model2/k20.pz", "model2/k20.pw_z")
    topicos = [1, 6]
    total, target = at.adjetive_index(
        load_file(dir_in, 'both_politics.txt'), at.topics, topics)
    index = (sum(target.values()) / sum(total.values()))
    print(index)
    


    # print("Saving graph file")
    # nx.write_gml(t_graph, dir_in + "topics_graph.gml")
    # f = open(dir_in + "compare_topics.txt", 'w')
    # f.write(txt)
    # f.write("\n\n-- political topics -- \n\n")
    # f.write(list2text(p_topics))
    # f.write("\n\n-- non_political topics -- \n\n")
    # f.write(list2text(np_topics))
    # f.write("\n\n-- both topics -- \n\n")
    # f.write(list2text(a_topics))
    # f.close()

    # # Separate by group
    # l, r = nx.bipartite.sets(t_graph)
    # pos = {}

    # # Update position for node from each group
    # pos.update((node, (1, index)) for index, node in enumerate(l))
    # pos.update((node, (2, index)) for index, node in enumerate(r))

    # nx.draw(B, pos=pos)
    # plt.show()
