import json
from pymongo import MongoClient
import configparser
import os
from text_processor import TextProcessor
    
cf = configparser.ConfigParser()
cf.read("file_path.properties")
path = dict(cf.items("file_path"))
dir_in = path['dir_in']
db_host = dict(cf.items("db_host"))
server = db_host['server']
port = int(db_host['port'])

client = MongoClient(server, port)
db = client.twitterdb
    
tw_files = sorted([file for root, dirs, files in os.walk(dir_in)
             for file in files if file.endswith('.json')])
tp = TextProcessor()
for tw_file in tw_files:
    with open(dir_in+tw_file) as data_file:
        for line in data_file:
            tweet = json.loads(line)
            tweet['text_processed'] = tp.text_process([tweet['text']], text_only=True)[0]   
            tweet['cond_55'] = 'reeleitos'
            db.tweets.insert(tweet)



