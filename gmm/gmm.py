import os
from gensim.models.word2vec import Word2Vec
import numpy as np
import sklearn.mixture as mix
import configparser
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import math


def make_dataset(model):
    """Make dataset from pre-trained Word2Vec model.

    Paramters
    ---------
    model: gensim.models.word2vec.Word2Vec
        pre-traind Word2Vec model as gensim object.

    Returns
    -------
    numpy.ndarray((vocabrary size, vector size))
        Sikitlearn's X format.
    """
    V = model.index2word
    X = np.zeros((len(V), model.vector_size))

    for index, word in enumerate(V):
        X[index, :] += model[word]
    return X

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

def purity_score(clusters, classes):
    A = np.c_[(clusters, classes)]
    n_accurate = 0.
    for j in np.unique(A[:, 0]):
        z = A[A[:, 0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])
    return n_accurate / A.shape[0]


def coherence_value(words, docs):
    # Test if each word w and w+1 in a topic are present in doc
    # Better if the value is near by 0
    c_sum = 0
    for i in range(len(words) - 1):
        presence = 0
        presence_both = 0
        for doc in docs:
            if (words[i + 1] in doc):
                presence += 1
                if (words[i] in doc):
                    presence_both += 1
        c_sum += math.log2((presence_both + 1) / presence)
    return c_sum

def topic_coherence(topics, docs, n):
    coherence_l = list()
    for tp in topics:
        n_words = tp[:n]
        coherence = coherence_value(n_words, docs)
        coherence_l.append(coherence)
    return coherence_l

if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_w2v = dir_w2v + "gmm/"
    dir_dataset = path['dir_dataset']
    k=20

    texts = list()
    txt = open(dir_dataset + "sanders_twitter.txt", "r")
    for l in txt:
        texts.append(l.split())

    #sentences = MySentences(dir_w2v)  # a memory-friendly iterator
    model = Word2Vec(texts , min_count = 1, workers = 4, size = 100, window = 11)
    model.save(dir_w2v + "model_word2vec")
    # model = gensim.models.Word2Vec.load(dir_w2v+"model_word2vec")

    print("Reading topics from file")
    labels = list()
    tw = open(dir_dataset + "topic_sanders_twitter.txt", "r")
    for l in tw:
        labels.append(l)
    i = 0
    num_topics = dict()
    for tp in set(labels):
        num_topics[tp] = i
        i = i + 1

    labels = [num_topics[x] for x in labels]
    words = [w for w, v in model.vocab.items()]
    word_vectors = model.syn0
    gmm_model = mix.GMM(n_components = k, n_iter = 1000, covariance_type = 'diag')
    gmm_model.fit(word_vectors)

    log_probs = mix.log_multivariate_normal_density(word_vectors, gmm_model.means_, gmm_model.covars_,
                                                gmm_model.covariance_type)
    word_topic = list()
    _, num_col = log_probs.shape
    for col in range(num_col):
        top_n = 10
        log_component_probs = (log_probs[:, col]).T
        sorted_indexes = np.argsort(log_component_probs)[::-1][:top_n]
        ordered_word_probs = [(model.index2word[idx], log_component_probs[idx]) for idx in sorted_indexes]
        word_topic.append([model.index2word[idx] for idx in sorted_indexes])

        print('---')
        print("Topic {0}".format(col + 1))
        print("Total prob:" + str(sum(log_component_probs)))
        print(", ".join(["{w}: {p}".format(w = w, p = p) for w, p in ordered_word_probs]))


    print("Assign topic to each text")
    prob = gmm_model.predict_proba(word_vectors)
    assign_topics = list()
    tw_l = list()
    dist_topics = list()
    for t in texts:
        tw_topics = list()
        lista = list()
        for tp in range(k - 1):
            tmp = 0
            tw = dict()
            for w in t:
                if w in model.vocab:
                    tw[w] = prob[model.vocab[w].index][tp]
                    tmp += prob[model.vocab[w].index][tp]
            lista.append(tw)
            tw_topics.append(tmp)
        dist_topics.append(tw_topics)
        tw_l.append(lista)
        assign_topics.append(tw_topics.index(max(tw_topics)))

    clf = svm.SVC(kernel = 'linear', C = 1)
    scores = cross_val_score(clf, dist_topics, labels, cv = k - 1)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(clf, dist_topics, labels, cv = k - 1, scoring = 'f1_macro')
    print("F1_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print("Topic Coherence")
    print(topic_coherence(word_topic, texts, 15))

    print("Topic NMI")
    print(normalized_mutual_info_score(assign_topics, labels))

    print("Topic Purity")
    print(purity_score(assign_topics,labels))





