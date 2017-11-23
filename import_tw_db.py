import json
import pymongo
import configparser
import os
from text_processor import TextProcessor
import xlrd
cf = configparser.ConfigParser()
cf.read("file_path.properties")
path = dict(cf.items("file_path"))
dir_in = path['dir_in']
dir_xls = path['dir_xls']

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client.twitterdb
tp = TextProcessor()
tw_files = sorted([file for root, dirs, files in os.walk(dir_in)
        for file in files if file.endswith('.json')])
excel = True

if excel:

    sheet_name = "nao_eleitos"
    col = 4
    rep_dic = {}
    for fname in tw_files:
        rep_dic[fname.split('_',1)[0]] = fname
    xls = xlrd.open_workbook(dir_xls)
    sheet = xls.sheet_by_name(sheet_name)
    for i in range(sheet.nrows):
        id_rep = str(int(sheet.cell_value(rowx= i, colx=col)))
        if (id_rep in rep_dic):
            with open(dir_in+rep_dic[id_rep]) as data_file:
                print('file %s' % data_file)
                for line in data_file:
                    tweet = json.loads(line)
                    tweet['text_processed'] = ' '.join(tp.text_process([tweet['text']], text_only=True)[0])
                    tweet['cond_55'] = sheet_name
                    db.tweets.insert(tweet)

else:                    
                
    for tw_file in tw_files:
        with open(dir_in+tw_file) as data_file:
            print('file %s' % data_file)
            for line in data_file:
                tweet = json.loads(line)
                tweet['text_processed'] = ' '.join(tp.text_process([tweet['text']], text_only=True)[0])
                tweet['cond_55'] = 'nao_eleitos'
                db.tweets.insert(tweet)
                