# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
from text_processor import TextProcessor
import configparser
import pickle
import numpy as np


def data_process(dir_in, file_in, dir_out, file_out):
    snippets = open(dir_in + file_in, "r")
    content = list()
    topic = list()
    for line in snippets:
        tmp = line.split(" ")
        content.append(tmp[:(len(tmp) - 1)])
        topic.append(tmp[-1])
    print("saving %s.pck " % file_out)
    with open(dir_out + file_out + ".pck", 'wb') as handle:
        pickle.dump(content, handle)
    print("saving %s.txt " % file_out)
    f = open(dir_out + file_out + ".txt", 'w+')
    for t in content:
        f.write(" ".join(t) + "\n")
    f.close
    print("saving %s_topic.txt " % file_out)
    f = open(dir_out + file_out + "_topic.txt", 'w+')
    for t in topic:
        f.write(topic)
    f.close

if __name__ == '__main__':

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_btm = path['dir_btm']
    dir_in = path['dir_in']
    dir_out = path['dir_out']
    dir_down = path['dir_down']
    tp = TextProcessor()
    data_process(dir_down, "data-web-snippets/train.txt",
                 dir_out, "snippets_train")
    data_process(dir_down, "data-web-snippets/test.txt",
                 dir_out, "snippets_test")
