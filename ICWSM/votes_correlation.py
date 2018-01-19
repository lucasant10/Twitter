import sys
sys.path.append('../')
import pymongo
import configparser
import os
import xlrd
from scipy import stats
from political_classification import PoliticalClassification
from collections import defaultdict
import math

class DataIter:
    def __init__(self, consulta):
         self.collection = pymongo.Connection().db.collection
         self.cursor = self.collection.find({consulta}) #return all
         self.age = None
         self.gender = None
    def __iter__(self):
         return self
    def next(self):
        if self.cursor.hasNext():
            data = self.cursor.next()
            self.set_data(data)
            return self
        else:
            raise StopIteration

# election
p1 = (1396483200000, 1412294400000)
p2 = (1412294400000, 1443830400000)
# impeachment
p3 = (1459382400000, 1472601600000)
p4 = (1472601600000, 1490918400000)

cf = configparser.ConfigParser()
cf.read("../file_path.properties")
path = dict(cf.items("file_path"))
dir_out = path['dir_out']

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client.twitterdb

pc = PoliticalClassification('model_lstm.h5',
                             'dict_lstm.npy', 16)      

xls = xlrd.open_workbook(dir_out + 'icwsm/votos.xlsx')
sheet = xls.sheet_by_index(0)

p_election = defaultdict(int)
p_election = defaultdict(int)
periods = [p1, p2, p3, p4]
txt = ''
logtxt = ''
for i, period in enumerate(periods):
    p = defaultdict(list)
    np = defaultdict(list)
    rt = defaultdict(list)
    v = defaultdict(list)
    lg_v = defaultdict(list)
    for r in range(sheet.nrows):
        politics = 0
        non_politics = 0
        screen_name = str(sheet.cell_value(rowx=r, colx=0))
        votes = int(sheet.cell_value(rowx=r, colx=1))       
        print('getting data: %s' % screen_name)
        tw = db.tweets.find({"screen_name" : screen_name,'created_at': {'$gte': period[0], '$lt': period[1]},'cond_55': {'$exists': True}})
        print('separating tweets')
        cond = None
        for tweet in tw:
            cond = tweet['cond_55']
            if pc.is_political(tweet['text_processed']):
                politics += 1
            else:
                non_politics += 1
        if cond != None:
            p[cond].append(politics)
            np[cond].append(non_politics)
            rt[cond].append((politics / (non_politics + politics)))
            v[cond].append(votes)
            lg_v[cond].append(math.log2(votes))

    for c in ['novos', 'nao_eleitos', 'reeleitos']:
        txt += '\n%s \ncondicao - pearsonr, p_value - spearman, p_value \n' % p1 
        txt += '%s - %0.2f, %0.2f - %0.2f, %0.2f \n' % (c, stats.pearsonr(p[c], v[c]), stats.spearman(p[c],v[c]))
        txt += '%s - %0.2f, %0.2f - %0.2f, %0.2f \n' % (c, stats.pearsonr(np[c], v[c]), stats.spearman(np[c],v[c]))
        txt += '%s - %0.2f, %0.2f - %0.2f, %0.2f \n' % (c, stats.pearsonr(rt[c], v[c]), stats.spearman(rt[c],v[c]))
    
    for c in ['novos', 'nao_eleitos', 'reeleitos']:
        logtxt += '\n%s \ncondicao - pearsonr, p_value - spearman, p_value \n' % p1 
        logtxt += '%s - %0.2f, %0.2f - %0.2f, %0.2f \n' % (c, stats.pearsonr(p[c], lg_v[c]), stats.spearman(p[c],lg_v[c]))
        logtxt += '%s - %0.2f, %0.2f - %0.2f, %0.2f \n' % (c, stats.pearsonr(np[c], lg_v[c]), stats.spearman(np[c],lg_v[c]))
        logtxt += '%s - %0.2f, %0.2f - %0.2f, %0.2f \n' % (c, stats.pearsonr(rt[c], lg_v[c]), stats.spearman(rt[c],lg_v[c]))

    f = open(dir_out + "vote_correlation.txt", 'w')
    f.write(txt)
    f.close()

    f = open(dir_out + "log_vote_correlation.txt", 'w')
    f.write(logtxt)
    f.close()