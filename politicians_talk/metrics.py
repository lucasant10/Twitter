import sys
import os
sys.path.append('../')
import configparser
import re
from read_twitter import ReadTwitter
from collections import Counter
import math, datetime
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import math

def tweet_week(date):
    dt_tw = pd.to_datetime(date * 1000000)
    return math.ceil((dt_tw - datetime.datetime(2013, 10, 4)).days / 7)

def cos_sim(parties, dic_features):
    tmp = dict()
    for party_i in parties:
        if len(dic_features) != 0:
            for party_j in parties:
                if party_i != party_j:
                    if party_i in tmp:
                        tmp[party_i] += cosine_similarity(dic_features[party_i], dic_features[party_j])[0][0]
                    else:
                        tmp[party_i] = cosine_similarity(dic_features[party_i], dic_features[party_j])[0][0]
        else:
            tmp[party_i] = 0
    if len(dic_features) != 0:
        s = sum(tmp.values())
        if(s!=0):
            #media de todas os cossenos
            tmp = {k : (v/s) for k, v in tmp.items()}

    return tmp

def features(w, dic_tweets):
    features = set()
    for k, v in dic_tweets.items():
        if k.split("_")[1] == str(w):
            features |= set(v.keys())
    return features

def vector_features(w, features, parties, dic_tweets):
    vec = dict()
    for ft in features:
        for party in parties:
            party_week = party + "_" + str(w)
            if party_week in dic_tweets:
                if party in vec:
                    vec[party].append(dic_tweets[party_week][ft])
                else:
                    vec[party] = [dic_tweets[party_week][ft]]
            else:
                if party in vec:
                    vec[party].append(0)
                else:
                    vec[party] = [0]
    return vec

def similarity(w, features, parties, dic_tweets):
    vector = vector_features(w, features,parties ,dic_tweets)
    return cos_sim(parties, vector)

def cultural_focus(w, features, parties, dic_tweets):
    tmp = dict()
    for party in parties:
        party_week = party + "_" + str(w)
        if party_week in dic_tweets:
            l = len(dic_tweets[party_week])
            if(l != 0):
                v = dic_tweets[party_week].values()
                a = [x / sum(v) for x in v]
                eps = 1e-15
                soma =  - sum([(n * math.log2(n + eps)) for n in a])
                if soma > 0 :
                    tmp[party] = 1 - (soma / math.log2(l + eps))
                else:
                    tmp[party] = 0
            else:
                tmp[party] = 0
        else:
            tmp[party] = 0
    return tmp


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']
    dir_out = path['dir_out']
    excel_path = path['excel_path']

    sheet_name = "amostra"
    col = 4
    rt = ReadTwitter(dir_in, excel_path, sheet_name, col)
    id_screenname_party = rt.id_screenname_party_from_xls()
    weeks = math.ceil((datetime.datetime(2015, 10, 4) - datetime.datetime(2013, 10, 4)).days / 7)

    hashtags = dict()
    mentions = dict()
    retweets = dict()
    dep_party = dict()
    for idx in range(len(id_screenname_party)):
        if id_screenname_party[idx][2] in dep_party:
            dep_party[id_screenname_party[idx][2]].append(id_screenname_party[idx][1])
        else:
            dep_party[id_screenname_party[idx][2]] = [id_screenname_party[idx][1]]

    fnames = ([file for root, dirs, files in os.walk(dir_out)
               for file in files if file.endswith('.pck')])
    if(("hashtags_counter.pck" and "mentions_counter.pck" and "retweets_counter.pck") in fnames):

        with open(dir_out + "hashtags_counter.pck", 'rb') as handle:
            hashtags = pickle.load(handle)

        with open(dir_out + "mentions_counter.pck", 'rb') as handle:
            mentions = pickle.load(handle)

        with open(dir_out + "retweets_counter.pck", 'rb') as handle:
            retweets = pickle.load(handle)
    else:
        for idx in range(len(id_screenname_party)):
            tweet = rt.tweets_election_data(str(id_screenname_party[idx][0]))
            for dt, tw in tweet.items():
                t_hashtags = re.findall(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", tw)
                t_mentions = re.findall(r'RT @\S+', tw)
                t_retweets = re.findall(r'(?<!RT\s)@\S+', tw)
                w = tweet_week(dt)
                party_week = id_screenname_party[idx][2] + "_" + str(w)
                if (party_week) in hashtags:
                    # caso exista o partido no dict adiciona ao counter existente
                    hashtags[party_week] += Counter(t_hashtags)
                    mentions[party_week] += Counter(t_mentions)
                    retweets[party_week] += Counter(t_retweets)
                else:
                    # cria o counter para o partido
                    hashtags[party_week] = Counter(t_hashtags)
                    mentions[party_week] = Counter(t_mentions)
                    retweets[party_week] = Counter(t_retweets)

        with open(dir_out + "hashtags_counter.pck", 'wb') as handle:
            pickle.dump(hashtags, handle)

        with open(dir_out + "mentions_counter.pck", 'wb') as handle:
            pickle.dump(mentions, handle)

        with open(dir_out + "retweets_counter.pck", 'wb') as handle:
            pickle.dump(retweets, handle)

    weeks_h_sim = dict()
    weeks_m_sim = dict()
    weeks_r_sim = dict()
    weeks_h_cf = dict()
    weeks_m_cf = dict()
    weeks_r_cf = dict()

    for w in range(1, weeks+1):
        m_features = features(w, mentions)
        weeks_m_sim[w] = similarity(w, m_features, dep_party.keys(), mentions)
        weeks_m_cf[w] = cultural_focus(w, features, dep_party.keys(), mentions)

        h_features = features(w, hashtags)
        weeks_h_sim[w] = similarity(w, h_features, dep_party.keys(), hashtags)
        weeks_h_cf[w] = cultural_focus(w, features, dep_party.keys(), hashtags)


        r_features = features(w, retweets)
        weeks_r_sim[w] = similarity(w, r_features, dep_party.keys(), retweets)
        weeks_r_cf[w] = cultural_focus(w, features, dep_party.keys(), retweets)


    hashtag_set = set()
    mention_set = set()
    retweet_set = set()

    for party_week in hashtags.keys():
        hashtag_set |= set(hashtags[party_week].keys())
        mention_set |= set(mentions[party_week].keys())
        retweet_set |= set(retweets[party_week].keys())
    features = hashtag_set
    features |= mention_set
    features |= retweet_set
    party_h_vec = dict()
    party_m_vec = dict()
    party_r_vec = dict()

    for ft in features:
        for party_week in hashtags.keys():
            party = party_week.split("_")[0]
            if party in party_h_vec:
                party_h_vec[party].append(hashtags[party_week][ft])
                party_m_vec[party].append(mentions[party_week][ft])
                party_r_vec[party].append(retweets[party_week][ft])
            else:
                party_h_vec[party] = [hashtags[party_week][ft]]
                party_m_vec[party] = [mentions[party_week][ft]]
                party_r_vec[party] = [retweets[party_week][ft]]

    cult_sim_h = dict()
    cult_sim_m = dict()
    cult_sim_r = dict()



#hashtags_list = sorted(list(hashtags.items()), key = lambda x: x[0].split("_")[1])
    #mentions_list = sorted(list(mentions.items()), key = lambda x: x[0].split("_")[1])
    #retweets_list = sorted(list(retweets.items()), key = lambda x: x[0].split("_")[1])
