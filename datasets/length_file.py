import sys
sys.path.append('../')
import configparser
from text_processor import TextProcessor
import topic_BTM as btm
import numpy as np
import math
import csv
import os

def return_files(extesion, path):
    fnames = ([file for root, dirs, files in os.walk(path)
               for file in files if file.endswith(extesion)])
    return fnames

if __name__ == '__main__':
    
    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_dataset = path['dir_dataset']


    files = return_files(".txt", dir_dataset + "data/")
    texto = "Número médio de palavras por arquivo \n"
    for f in files:
        print("Reading file %s" % f)
        txt = open(dir_dataset + "data/" + f, "r")
        tam = 0
        qtd = 0
        for l in txt:
            tam += len(l.split())
            qtd += 1
        texto += f + ": %0.2f \n" % (tam/qtd)
    
    print("Saving file")
    f = open(dir_dataset + "data/" + "num_palavras.txt", 'w+')
    f.write(texto)
    f.close()