import sys
sys.path.append('../')
import configparser
import json
import os
from collections import defaultdict
import math
import pymongo
import numpy as np
from political_classification import PoliticalClassification
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == "__main__":
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']

    pc = PoliticalClassification('cnn_s300.h5', 'cnn_s300.npy', 18)

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    tweets = db.tweets.find({'created_at': {'$gte': 1380585600000, '$lt': 1443830400000}, 'cond_55': {'$exists': True}})

    pol = ''
    n_pol = ''
    for tw in tweets:
        tweet = tw['text_processed']
        if pc.is_political(tweet):
            pol += tweet + '\n'
        else:
            n_pol += tweet + '\n'

    f =  open(dir_in + "CSCW/politics.txt", 'w')
    f.write(pol)
    f.close()

    f =  open(dir_in + "CSCW/non_politics.txt", 'w')
    f.write(n_pol)
    f.close()


