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
import pandas as pd
from burst import Burst
import numpy as np



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
        if (s != 0):
            # media de todas os cossenos
            tmp = {k: (v / s) for k, v in tmp.items()}

    return tmp


def features(w, dic_tweets):
    features = set()
    for k, v in dic_tweets.items():
        if k.split("_")[1] == str(w):
            features |= set(v.keys())
    return features

def party_features(party, dic_tweets):
    features = set()
    for w in range(1, weeks + 1):
        party_week = party + "_" + str(w)
        if party_week in dic_tweets:
                features |= set(dic_tweets[party_week])
    return features


def ranked_features(w, party, dic_tweets):
    features = set()
    party_week = party + "_" + str(w)
    if party_week in dic_tweets:
        features |= set(sorted(dic_tweets[party_week], key = dic_tweets[party_week].get, reverse = True))
    return list(features)


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


def hIndex(citations):
    citations.sort(reverse = True)
    return max([min(k + 1, v) for k, v in enumerate(citations)]) if citations else 0


def similarity(w, features, parties, dic_tweets):
    vector = vector_features(w, features, parties, dic_tweets)
    return cos_sim(parties, vector)


def cultural_focus(w, parties, dic_tweets):
    tmp = dict()
    for party in parties:
        party_week = party + "_" + str(w)
        if party_week in dic_tweets:
            l = len(dic_tweets[party_week])
            if (l != 0):
                v = dic_tweets[party_week].values()
                a = [x / sum(v) for x in v]
                eps = 1e-15
                soma = - sum([(n * math.log2(n + eps)) for n in a])
                if soma > 0:
                    tmp[party] = 1 - (soma / math.log2(l + eps))
                else:
                    tmp[party] = 0
            else:
                tmp[party] = 0
        else:
            tmp[party] = 0
    return tmp


def cultural_reproduction(w, f_week_1, f_week_2, p):
    d = min(len(f_week_1), len(f_week_2))
    s = 0
    for i in range(1, d + 1):
        s += (2 * len(set(f_week_1[:i]).intersection(f_week_2[:i])) /
              (len(f_week_1[:i]) + len(f_week_1[:i]))) * math.pow(p, (i - 1))
    return (1 - p) * s


def c_reproduction(w, party, dic_tweets):
    f_week_1 = ranked_features(w, party, dic_tweets)
    f_week_2 = ranked_features(w + 1, party, dic_tweets)
    return cultural_reproduction(w, f_week_1, f_week_2, p)

def party_h_index(features, weeks, party, dic_tweets):
    f_index = dict()
    for f in features:
        tmp = list()
        for w in range(1, weeks + 1):
            party_week = party + "_" + str(w)
            if party_week in dic_tweets:
                tmp.append(dic_tweets[party_week][f])
            else:
                tmp.append(0)
        f_index[f] = hIndex(tmp)
    return f_index

def f_gamma(s, r, d, qtd_r, qtd_d):
    p = 0
    if (len(qtd_d) != 0):
        p = sum(qtd_r) / sum(qtd_d) * (math.pow(2, s))
    print("s %d,r %d,d %d,R %d, D %d" % (s,r,d,sum(qtd_r),sum(qtd_d)))
    tmp = misc.comb(d, r) * (math.pow(p, r)) * (math.pow((1 - p), (d - r)))
    if tmp != 0:
        return (- math.log(tmp))
    return tmp

def party_burstiness(features, weeks, party, dic_tweets):
    bt = Burst()
    b_index = dict()
    for f in features:
        qtd_r = list()
        qtd_d = list()
        for w in range(1, weeks + 1):
            party_week = party + "_" + str(w)
            if party_week in dic_tweets:
                qtd_r.append(dic_tweets[party_week][f])
                qtd_d.append(sum(dic_tweets[party_week].values()))
            else:
                qtd_r.append(0)
                qtd_d.append(0)
        n = sum(qtd_r)
        r = np.asarray(qtd_r)
        d = np.asarray(qtd_d)
        q, d, r, p = bt.burst_detection(r, d, n, 2, 1, 2)
        bursts = bt.enumerate_bursts(q, 'burstLabel')
        b_index[f] = bursts
    return b_index

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
    if (("hashtags_counter.pck" and "mentions_counter.pck" and "retweets_counter.pck") in fnames):

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

    print("\nCosine Similarity\n")
    for w in range(1, weeks + 1):
        m_features = features(w, mentions)
        weeks_m_sim[w] = similarity(w, m_features, dep_party.keys(), mentions)
        h_features = features(w, hashtags)
        weeks_h_sim[w] = similarity(w, h_features, dep_party.keys(), hashtags)
        r_features = features(w, retweets)
        weeks_r_sim[w] = similarity(w, r_features, dep_party.keys(), retweets)

    print("\nCultural Focus\n")
    for w in range(1, weeks + 1):
        weeks_h_cf[w] = cultural_focus(w, dep_party.keys(), hashtags)
        weeks_m_cf[w] = cultural_focus(w, dep_party.keys(), mentions)
        weeks_r_cf[w] = cultural_focus(w, dep_party.keys(), retweets)

    print("\nCultural Reproduction\n")
    p = 0.9
    party_h_cr = dict()
    party_m_cr = dict()
    party_r_cr = dict()
    for w in range(1, weeks):
        for party in dep_party.keys():
            party_week = party + "_" + str(w)
            party_h_cr[party_week] = c_reproduction(w, party, hashtags)
            party_m_cr[party_week] = c_reproduction(w, party, mentions)
            party_r_cr[party_week] = c_reproduction(w, party, hashtags)

    print("\nInstitutionness\n")
    party_h_hi = dict()
    party_m_hi = dict()
    party_r_hi = dict()
    dict_features = dict()
    for party in dep_party.keys():
        h_features = party_features(party, hashtags)
        party_h_hi[party] = party_h_index(h_features, weeks, party, hashtags)
        m_features = party_features(party, mentions)
        party_m_hi[party] = party_h_index(m_features, weeks, party, mentions)
        r_features = party_features(party, retweets)
        party_r_hi[party] = party_h_index(r_features, weeks, party, retweets)

    print("\nBurstiness\n")
    party_h_bu = dict()
    party_m_bu = dict()
    party_r_bu = dict()
    for party in dep_party.keys():
        h_features = party_features(party, hashtags)
        party_h_bu[party] = party_burstiness(h_features, weeks, party, hashtags)
        m_features = party_features(party, mentions)
        party_m_bu[party] = party_burstiness(m_features, weeks, party, mentions)
        r_features = party_features(party, retweets)
        party_r_bu[party] = party_burstiness(r_features, weeks, party, retweets)
