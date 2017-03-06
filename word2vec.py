import gensim, logging
import os
import configparser
import numpy as np
from matplotlib import pyplot as plt
from sklearn import manifold, datasets


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

if __name__=='__main__':

cf = configparser.ConfigParser()
cf.read("file_path.properties")
path = dict(cf.items("file_path"))
dir_w2v = path['dir_w2v']

    sentences = MySentences(dir_w2v) # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences, min_count=10, workers=4,size=200)
    model.save(dir_w2v+"model_word2vec")
    #model = gensim.models.Word2Vec.load(dir_w2v+"model_word2vec")

    words = [w for w,v in model.vocab.items() if v.count>300]
    vectors = [model[l] for l in words]

tsne = manifold.TSNE(n_components=2, random_state=0)
Y = tsne.fit_transform(vectors)
plt.scatter(vis_x, vis_y, c=colors, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.show()
plt.clf()
plt.scatter(Y[:, 0], Y[:, 1],c=np.random.rand(3,5),cmap=plt.cm.Spectral)
plt.show()

colors = cm.rainbow(np.linspace(0, 1, len(vectors[0])))


