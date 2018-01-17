import sys
sys.path.append('../')
import configparser
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from random import shuffle
from collections import defaultdict
import pymongo

if __name__ == '__main__':

    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    deps = db.tweets.aggregate([
                                {'$match': {'created_at': {'$gte': 1380585600000, '$lt': 1506816000000},
                                            'cond_55': {'$exists': True}}},
                                {'$group': {'_id': {'user': '$user_id', 'cond': '$cond_55'},
                                            "texts": {'$push': "$text_processed"}}}], allowDiskUse=True)

    sentences = list()
    label = {'reeleitos': 'REELECTED',
             'nao_eleitos': 'NOT_REELECTED', 'novos': 'NEWCOMERS'}
    count = defaultdict(int)
    for dep in deps:
        text = ' '.join(dep['texts'])
        sentences.append(LabeledSentence(
            text.split(), [label[dep['_id']['cond']] + '_%s' % count[dep['_id']['cond']]]))
        count[dep['_id']['cond']] += 1

    model = Doc2Vec(min_count=1, window=10, size=300,
                    sample=1e-4, negative=5, workers=8, iter=20)
    model.build_vocab(sentences)
    print('trainning')
    model.train(sentences, total_examples = model.corpus_count, epochs = model.iter)
    
    model.save(dir_w2v + "model_doc2vec.d2v")
