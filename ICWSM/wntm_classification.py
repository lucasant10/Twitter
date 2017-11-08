import sys
sys.path.append('../')
import configparser
import gensim
import numpy as np



if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_lda = path['dir_lda']

    model = gensim.models.LdaModel.load(dir_lda+filename)

    print("Loading baseline file " )
    baseline = list()
    tweets = open(dir_out + "tweets_baseline.txt", "r")
    for l in tweets:
        baseline.append(l.split())