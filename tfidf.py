import math
from collections import Counter
from PtBrTwitter import PtBrTwitter 
import json
import os
from text_processor import TextProcessor
import itertools
import pickle


class TfIdf():

    tp = TextProcessor()

    def tf(self, word, w_counter):
        return w_counter[word] / len(w_counter)

    def n_containing(self, word, doc_counter):
        return sum(1 for w_counter in doc_counter if word in w_counter)

    def idf(self, word, doc_counter):
        return math.log(len(doc_counter) / (1 + self.n_containing(word, doc_counter)))

    def tfidf(self, word, w_counter, doc_counter):
        return self.tf(word, w_counter) * self.idf(word, doc_counter)

    def read_files(self, dir_in):
        doc_list=list()
        count_list=list()
        tw_files = ([file for root, dirs, files in os.walk(dir_in)
            for file in files if file.endswith('.json') ])
        for tw_file in tw_files:
            with open(dir_in+tw_file) as data_file:
                for line in data_file:
                    tweet = json.loads(line)
                    doc_list.append(tweet['text'])
            tweets = self.tp.text_process(doc_list)
            tweets = list(itertools.chain.from_iterable(tweets))
            count_list.append(Counter(tweets))

        return tw_files, count_list

    def save_counter(self, dir_in,dir_out):
        doc_list=list()
        tw_files = ([file for root, dirs, files in os.walk(dir_in)
            for file in files if file.endswith('.json') ])
        # lista os arquivos ja processados
        pck = ([file for root, dirs, files in os.walk(dir_out)
            for file in files if file.endswith('.pck') ])

        tw_files = [x for x in tw_files if x.split('.')[0]+".pck" not in pck]
        for i,tw_file in enumerate(tw_files):
            print("processando o arquivo: "+tw_files[i]+"\n")
            with open(dir_in+tw_file) as data_file:
                for line in data_file:
                    tweet = json.loads(line)
                    doc_list.append(tweet['text'])
            tweets = self.tp.text_process(doc_list)
            tweets = list(itertools.chain.from_iterable(tweets))
            t_count = Counter(tweets)
            with open(dir_out+tw_files[i].split('.')[0]+".pck", 'wb') as handle:
                pickle.dump(t_count, handle)


if __name__=='__main__':

    dir_in = "/Users/lucasso/Documents/tweets_pedro/"
    dir_out = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/Tf-Idf/Counter_dep/"
    ptbr = PtBrTwitter(dir_in,dir_out)
    tfidf = TfIdf()
    tfidf.save_counter(dir_in,dir_out)


    """
    for c in count_list:
        tot_counter += c

    for i, w_counter in enumerate(count_list):
        dict_term = {word: tfidf.tfidf(word, w_counter, count_list) for word in tot_counter}
        dic = {k:v for k,v in dict_term.items() if v<0}
        sort = sorted(dic.items(), key=lambda x: x[1])
        with open(dir_out+tw_files[i]+".pck", "w") as f: f.write(json.dumps(sort))
    """

    







    """
    # TfIdf dos Deputados 
    round_c = ptbr.loadCounter("Counter_pt-br.pck")
    dep_c = ptbr.loadCounter("Counter_Dep.pck")
    doc_counter = [dep_c, round_c]
    tot_counter = Counter()
    dict_term = dict()

    for c in doc_counter:
        tot_counter += c

    for i, w_counter in enumerate(doc_counter):

        dict_term = {word: tfidf.tfidf(word, w_counter, doc_counter) for word in tot_counter}
        dic = {k:v for k,v in dict_term.items() if v<0}
        sort = sorted(dic.items(), key=lambda x: x[1])
        with open(dir_out+"doc_"+str(i)+".txt", "w") as f: f.write(json.dumps(sort))

    """
        
    