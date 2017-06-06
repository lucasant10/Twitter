import gensim
import logging
import os
import configparser
import numpy as np
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from collections import namedtuple


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

if __name__ == '__main__':

cf = configparser.ConfigParser()
cf.read("file_path.properties")
path = dict(cf.items("file_path"))
dir_w2v = path['dir_w2v']
dir_w2v = dir_w2v + "2k_like/"

sentences = MySentences(dir_w2v)  # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences, min_count=7, workers=4, size=100)
model.save(dir_w2v + "model_word2vec")
#model = gensim.models.Word2Vec.load(dir_w2v+"model_word2vec")

words = [w for w, v in model.vocab.items() if v.count > 100]
#words = [w for w,v in model.vocab.items() ]
vectors = [model[l] for l in words]

tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=50)
Y = tsne.fit_transform(vectors)
plt.clf()
plt.scatter(Y[:, 0], Y[:, 1], c=np.random.rand(3, 4), cmap=plt.cm.Spectral)
for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.show()

colors = cm.rainbow(np.linspace(0, 1, len(vectors[0])))

# Doc2Vec
pck = ([file for root, dirs, files in os.walk(dir_w2v)
        for file in files if file.endswith('.txt')])

doc_list = list()

for m in pck:
    dp = []
    with open(dir_w2v + m, "rb") as data_file:
        for line in data_file:
            dp.append(line.decode('utf-8'))
    doc_list.append(dp)

analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
docs = list()
for i, text in enumerate(doc_list):
    words = " ".join(text).split()
    tags = [i]
    docs.append(analyzedDocument(words, tags))

model = Doc2Vec(docs, min_count=7, workers=4, size=100)
model.save(dir_w2v + "model_doc2vec")
#model = gensim.models.Word2Vec.load(dir_w2v+"model_word2vec")
vectors = [l for l in model.docvecs]

tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=50)
Y = tsne.fit_transform(vectors)
plt.clf()
plt.scatter(Y[:, 0], Y[:, 1], c=np.random.rand(3, 4), cmap=plt.cm.Spectral)
plt.show()
