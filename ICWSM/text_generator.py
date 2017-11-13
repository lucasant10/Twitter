import sys
sys.path.append('../')
import argparse
import configparser
from text_processor import TextProcessor
import json
import os

def load_files(dir_in):
    doc_list = list()
    tw_files = sorted([file for root, dirs, files in os.walk(dir_in)
                 for file in files if file.endswith('.json')])
    for tw_file in tw_files:
        temp = list()
        with open(dir_in+tw_file) as data_file:
            for line in data_file:
                tweet = json.loads(line)
                temp.append(tweet['text'])
        doc_list.append(temp)
    return doc_list, tw_files


if __name__ == "__main__":
    
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_val']

    tp = TextProcessor()
    tweets = list()
    doc_list, tw_files = load_files(dir_in)
    for txt in doc_list:
        print(len(doc_list))
        tweets.append(tp.text_process(txt, text_only=True))

    for i, fl in enumerate(tw_files):
        f =  open(dir_in+"%s.txt" % fl.split('.')[0], 'w')
        for tw in tweets[i]:
            f.write(" ".join(tw) + "\n")

        f.close()


