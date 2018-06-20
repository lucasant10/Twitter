import sys
sys.path.append('../')
import configparser
import topic_BTM as btm
from collections import Counter, defaultdict
from assign_topics import AssingTopics
import pymongo

def vocab(path):
    voca = btm.read_voca(dir_btm + path)
    inv_voca = {v: k for k, v in voca.items()}
    return voca, inv_voca

if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_btm = path['dir_btm']
    dir_in = path['dir_in']

    person_topicos = [4, 6, 7, 9, 12, 13, 14, 16, 18]

    print("Reading vocab ")
    all_voca, all_inv_voca = vocab('all_voca2.txt')

    assing_topics = list()
    print("Reading topic")
    a_pz_pt = dir_btm + "model2/k20.pz"
    a_pz = btm.read_pz(a_pz_pt)
    a_zw_pt = dir_btm + "model2/k20.pw_z"

    print("processing assign topic distribution")
    at = AssingTopics(dir_btm, dir_in, 'all_voca2.txt',
                      "model2/k20.pz", "model2/k20.pw_z")

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    tweets = db.tweets.find({'created_at': {
                            '$gte': 1380596400000, '$lt': 1443668400000}, 'cond_55': {'$exists': True}})

    topic_tweets = defaultdict(list)
    for tw in tweets:
        tweet=tw['text_processed'].split()
        index=at.get_text_topic(tweet, at.topics)
        topic_tweets[index].append(tw['text_processed'])

    for topic, tws in topic_tweets.items():
        with open(dir_in + 'tw_topics/topic_%s.txt' % topic, 'a') as f:
            for item in tws:
                f.write("%s\n" % item)
        f.close()
