import sys
sys.path.append('../')
import configparser
import gensim
import numpy as np
from sklearn.metrics import classification_report


if __name__ == '__main__':

    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_lda = path['dir_lda']
    dir_in = path['dir_in']

    k =2
    print("processing model %sk " % k)
    model = gensim.models.LdaModel.load(dir_lda + "wntm_model_%sk.lda" % k)
    vocab = model.id2word

    print("Loading tweets_politicos file ")
    bow_politics = list()
    tweets = open(dir_in + "tweets_politicos.txt", "r")
    for l in tweets:
        bow_politics.append(vocab.doc2bow(l.split()))

    print("Loading tweets_nao_politicos file ")
    bow_n_politics = list()
    tweets = open(dir_in + "tweets_nao_politicos.txt", "r")
    for l in tweets:
        bow_n_politics.append(vocab.doc2bow(l.split()))

    print("Assing politics topics")
    y_pred = list()
    y_true = list()
    for txt in bow_politics:
        y_pred.append(np.argmax([x[1] for x in model[txt]]))
        y_true.append(0)


    print("Assing not politics topics")
    assing_n_politcs = list()
    for txt in bow_n_politics:
        y_pred.append(np.argmax([x[1] for x in model[txt]]))
        y_true.append(1)

    print(classification_report(y_true, y_pred))
   
