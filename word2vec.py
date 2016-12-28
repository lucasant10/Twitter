import gensim, logging
import os
import configparser
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
    model = gensim.models.Word2Vec.load(dir_w2v+"model_word2vec")

