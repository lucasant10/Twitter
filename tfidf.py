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

    @staticmethod
    def tf(word, w_counter):
        return (w_counter[word] / float(sum(w_counter.values())))

    @staticmethod
    def n_containing( word, doc_counter):
        count = 0
        for document in doc_counter:
            if document[word] > 0:
                count += 1
        return count

    @staticmethod
    def idf( word, doc_counter):
        return (math.log(len(doc_counter) / float(n_containing(word, doc_counter))))

    @staticmethod
    def tfidf( word, w_counter, doc_counter):
        return (self.tf(word, w_counter) * self.idf(word, doc_counter))

    def read_files(self, dir_in):
        count_list=list()
        tw_files = ([file for root, dirs, files in os.walk(dir_in)
            for file in files if file.endswith('.json') ])
        for tw_file in tw_files:
            doc_list=list()
            with open(dir_in+tw_file) as data_file:
                for line in data_file:
                    tweet = json.loads(line)
                    doc_list.append(tweet['text'])
            tweets = self.tp.text_process(doc_list)
            tweets = list(itertools.chain.from_iterable(tweets))
            count_list.append(Counter(tweets))

        return tw_files, count_list

    def save_counters(self, dir_in,dir_out):
        tw_files = ([file for root, dirs, files in os.walk(dir_in)
            for file in files if file.endswith('.json') ])
        # lista os arquivos ja processados
        pck = ([file for root, dirs, files in os.walk(dir_out)
            for file in files if file.endswith('.pck') ])

        tw_files = [x for x in tw_files if x.split('.')[0]+".pck" not in pck]
        for i,tw_file in enumerate(tw_files):
            print("processando o arquivo: "+tw_files[i]+"\n")
            doc_list=list()
            with open(dir_in+tw_file) as data_file:
                for line in data_file:
                    tweet = json.loads(line)
                    doc_list.append(tweet['text'])
            tweets = self.tp.text_process(doc_list)
            tweets = list(itertools.chain.from_iterable(tweets))
            t_count = Counter(tweets)
            with open(dir_out+tw_files[i].split('.')[0]+".pck", 'wb') as handle:
                pickle.dump(t_count, handle)

    def save_tfidf(self, dir_in,dir_out):
        counter_list = list()
        tot_counter = Counter()
        pck = ([file for root, dirs, files in os.walk(dir_in)
            for file in files if file.endswith('.pck') ])

        txt = ([file for root, dirs, files in os.walk(dir_out)
            for file in files if file.endswith('.txt') ])
        
        pck = [x for x in pck if x.split('.')[-1]+".txt" not in txt]

        for i,counter_file in enumerate(pck):
            print("processando o arquivo: "+counter_file+"\n")
            with open(dir_in+counter_file, 'rb') as data_file:
                tw_counter = pickle.load(data_file)
                tot_counter += tw_counter
                counter_list.append(tw_counter)
        for i, w_counter in enumerate(counter_list):
            print("gerando o tfidf de "+pck[i])
            dict_term = {word: self.tfidf(word, w_counter, counter_list) for word in tot_counter}
            dic = {k:v for k,v in dict_term.items() if v>0}
            sort = sorted(dic.items(), key=lambda x: x[1], reverse=True)
            print("salvando o arquivo "+pck[i])
            with open(dir_out+pck[i]+".txt", "w") as f:
             f.write(json.dumps(sort))
             f.close()
    def create_table_parl_tfidf():
        counter_list = list()
        tot_counter = Counter()
        pck = ([file for root, dirs, files in os.walk(dir_in)
            for file in files if file.endswith('.pck') ])
        for i,counter_file in enumerate(pck):
            print("processando o arquivo: "+counter_file+"\n")
            with open(dir_in+counter_file, 'rb') as data_file:
                tw_counter = pickle.load(data_file)
                tot_counter += tw_counter
                counter_list.append(tw_counter)

            

if __name__=='__main__':

    dir_in = "/Users/lucasso/Documents/tweets_pedro/"
    dir_out = "/Users/lucasso/Documents/pck/"
    ptbr = PtBrTwitter(dir_in,dir_out)
    tfidf = TfIdf()
    tfidf.save_counters(dir_in,dir_out)
    tfidf.save_tfidf(dir_in, dir_out)



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
        dic = {k:v for k,v in dict_term.items() if v>0}
        sort = sorted(dic.items(), key=lambda x: x[1])
        with open(dir_out+"doc_"+str(i)+".txt", "w") as f: f.write(json.dumps(sort))

    """
        
    