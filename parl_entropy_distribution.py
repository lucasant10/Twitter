from collections import Counter
import json
import os
from text_processor import TextProcessor
from collections import Counter
import itertools
import pickle
import configparser
from tfidf import TfIdf
import matplotlib.pyplot as plt



def load_counters(dir_path):
    doc_list=list()
    counters = ([file for root, dirs, files in os.walk(dir_path)
        for file in files if file.endswith('.pck') ])
    parl_counter_list = list()
    tot_counter = Counter()
    for c in counters:
        with open(dir_out+c,'rb') as handle:
            tw_counter = pickle.load(handle)
            parl_counter_list.append(tw_counter)
            tot_counter += tw_counter
    return tot_counter, parl_counter_list 




if __name__=='__main__':

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']
    dir_out = path['dir_out']
    dir_ale = path['dir_ale']
    dir_rob = path['dir_rob']

    tot_counter, parl_counter_list = load_counters(dir_out)
    tp = TextProcessor()
    tfidf = TfIdf()

    word_entropy = dict()
    for word in tot_counter:
        word_entropy[word] = tfidf.parl_entropy(word,tot_counter,parl_counter_list)

    freq = [int(math.pow(2,x)) for x in word_entropy.values() ]
plt.hist(freq, 15)
plt.xticks(np.arange(0,max(freq),20))
#plt.gca().set_yscale("log")
plt.xlabel("# de deputados que utilizaram a palavra" )
plt.ylabel("# palavras utilizadas pelos deputados" )
plt.show()
plt.clf()


    frequencies = {key:float(value)/sum(y.values()) for (key,value) in y.items()}

with open(dir_out+"word_entropy.pck", 'wb') as handle:
    pickle.dump(word_entropy, handle)

def reject_outliers(data):
    m = 1
    u = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (u - m * s < e < u + m * s)]
    return filtered


    
