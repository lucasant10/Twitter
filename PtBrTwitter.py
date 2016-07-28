from text_processor import TextProcessor
from collections import Counter
import itertools
import json
import os
import pickle


class PtBrTwitter():


    def __init__(self, dir_in, dir_out):
        self.dir_in = dir_in
        self.dir_out = dir_out
        self.tw_files = ([file for root, dirs, files in os.walk(self.dir_in)
            for file in files if file.endswith('.json') ])
        self.doc_list = list()
        self.date_list = list()
        self.tp = TextProcessor()

    def read(self):
        
        for tw_file in self.tw_files:
            with open(self.dir_in+tw_file) as data_file:
                for line in data_file:
                    tweet = json.loads(line)
                    self.doc_list.append(tweet['text'])
                    self.date_list.append(tweet['created_at'])
    def tokenizeAndSave(self, file_name):
        tweets = self.tp.text_process(self.doc_list)
        tweets = list(itertools.chain.from_iterable(tweets))
        t_count = Counter(tweets)
        with open(self.dir_out+file_name, 'wb') as handle:
            pickle.dump(t_count, handle)

    def loadCounter(self, file_name):
        with open(self.dir_out+file_name, 'rb') as handle:
            t_count = pickle.load(handle)
        return t_count

if __name__=='__main__':

    dir_in = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/"
    dir_out = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/plot/"
    ptbr = PtBrTwitter(dir_in,dir_out)
    ptbr.read()
    ptbr.tokenizeAndSave("Counter_Dep.pck")





       




           
        
        