import json
import pymongo
import configparser
import os
from text_processor import TextProcessor
    
cf = configparser.ConfigParser()
cf.read("file_path.properties")
path = dict(cf.items("file_path"))
dir_in = path['dir_in']

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client.twitterdb

tw_files = sorted([file for root, dirs, files in os.walk(dir_in)
             for file in files if file.endswith('.json')])
tp = TextProcessor()
for tw_file in tw_files:
    with open(dir_in+tw_file) as data_file:
        print('file %s' % data_file)
        for line in data_file:
            tweet = json.loads(line)
            tweet['text_processed'] = ' '.join(tp.text_process([tweet['text']], text_only=True)[0])
            tweet['cond_55'] = 'nao_eleitos'
            db.tweets.insert(tweet)
            
            



