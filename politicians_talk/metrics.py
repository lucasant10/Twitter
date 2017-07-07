import sys
import os

sys.path.append('../')
import configparser
import re
from read_twitter import ReadTwitter
from collections import Counter
import datetime
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import math
import pandas as pd
from burst import Burst
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import random





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
        r = np.array(qtd_r)
        d = np.array(qtd_d)
        n = len(r)
        q, d, r, p = bt.burst_detection(r, d, n, 2, 1, 2)
        bursts = bt.enumerate_bursts(q, 'burstLabel')
        b_index[f] = bursts
    return b_index

def plot_similarity(parties, weeks, dic_tweets, dic_color, f_name, y_title):
    data = list()
    for party in parties:
        tmp = list()
        for w in range(1, weeks + 1):
            tmp.append(dic_tweets[w][party])
        data.append(go.Scatter(
            y = tmp,
            x = [x for x in range(1, weeks)],
            mode = "lines+markers",
            marker = go.Marker(color = dic_color[party]),
            name = party,
            line = dict(color = dic_color[party]),
            opacity = 0.8))
    data.append(go.Scatter(
            x = [53,53],
            y = [0,1],
            mode = "lines",
            line = go.Line(color = "#111111", width = 2),
            showlegend = False
        )
    )

    layout = go.Layout(
        title = f_name.split("/")[-1],
        annotations = [
            dict(
                x = 53,
                y = 0.9,
                xref = 'x',
                yref = 'y',
                text = 'Semana da Eleição',
                showarrow = True,
                ax = 100,
                ay = -30,
                font = dict(
                    family = 'Courier New, monospace',
                    size = 16,
                    color = '#696969'
                )
            )
        ],
        xaxis = dict(
            title = 'Semanas de 04/10/2013 a 04/10/2015',
            nticks = 40,
            domain = [0, 1],
            titlefont = dict(
                family = 'Arial, sans-serif',
                size = 18,
                color = 'grey'
            )
        ),
        yaxis = dict(
            title = y_title,
            titlefont = dict(
                family = 'Arial, sans-serif',
                size = 18,
                color = 'grey'
            )
        )
    )
    fig = go.Figure(data = data, layout = layout)
    plot(fig, filename = f_name)

def plot_reproduction(parties, weeks, dic_tweets, dic_color, f_name):
        data = list()
        for party in parties:
            tmp = list()
            for w in range(1, weeks):
                party_week = party + "_" + str(w)
                tmp.append(dic_tweets[party_week])
            data.append(go.Scatter(
                y = tmp,
                x = [x for x in range(1, weeks)],
                mode = "lines+markers",
                marker = go.Marker(color = dic_color[party]),
                name = party,
                line = dict(color = dic_color[party]),
                opacity = 0.8))
        data.append(go.Scatter(
            x=[53, 53],
            y=[0, 1],
            mode="lines",
            line=go.Line(color="#111111", width=2),
            showlegend=False
        )
        )

        layout = go.Layout(
            title = f_name.split("/")[-1],
            annotations=[
                dict(
                    x=53,
                    y=0.9,
                    xref='x',
                    yref='y',
                    text='Semana da Eleição',
                    showarrow=True,
                    ax=100,
                    ay=-30,
                    font=dict(
                        family='Courier New, monospace',
                        size=16,
                        color='#696969'
                    )
                )
            ],
            xaxis = dict(
                title = 'Semanas de 04/10/2013 a 04/10/2015',
                nticks = 40,
                domain = [0, 1],
                titlefont = dict(
                    family = 'Arial, sans-serif',
                    size = 18,
                    color = 'grey'
                )
            ),
            yaxis = dict(
                title = 'Cultural Reproduction (RBO)',
                titlefont = dict(
                    family = 'Arial, sans-serif',
                    size = 18,
                    color = 'grey'
                )
            )
        )
        fig = go.Figure(data = data, layout = layout)
        plot(fig, filename = f_name)

if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']
    dir_out = path['dir_out']
    excel_path = path['excel_path']

    sheet_name = "new"
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

    colors = ['aqua', 'black', 'blue', 'fuchsia', 'gray', 'green',
              'lime', 'maroon', 'navy', 'olive', 'orange', 'purple', 'red',
              'silver', 'teal', 'cyan4', 'yellow','pink3','DarkOrchid',
              'tomato1','yellow3', 'turquoise','thistle','SlateGray2', 'LightCoral','green4' ]
    #"#%06x" % random.randint(0, 0xFFFFFF)
    dic_color = {v: colors[i]  for i, v in enumerate(dep_party.keys())}

    print("\nCosine Similarity\n")
    weeks_h_sim = dict()
    weeks_m_sim = dict()
    weeks_r_sim = dict()
    for w in range(1, weeks + 1):
        m_features = features(w, mentions)
        weeks_m_sim[w] = similarity(w, m_features, dep_party.keys(), mentions)
        h_features = features(w, hashtags)
        weeks_h_sim[w] = similarity(w, h_features, dep_party.keys(), hashtags)
        r_features = features(w, retweets)
        weeks_r_sim[w] = similarity(w, r_features, dep_party.keys(), retweets)

    plot_similarity(dep_party.keys(), weeks, weeks_h_sim, dic_color, dir_out + "similarity_hastags",
                    'Similaridade Cultural (Cosseno)')
    plot_similarity(dep_party.keys(), weeks, weeks_m_sim, dic_color, dir_out + "similarity_mentions",
                    'Similaridade Cultural (Cosseno)')
    plot_similarity(dep_party.keys(), weeks, weeks_r_sim, dic_color, dir_out + "similarity_retweets",
                    'Similaridade Cultural (Cosseno)')

    print("\nCultural Focus\n")
    weeks_h_cf = dict()
    weeks_m_cf = dict()
    weeks_r_cf = dict()
    for w in range(1, weeks + 1):
        weeks_h_cf[w] = cultural_focus(w, dep_party.keys(), hashtags)
        weeks_m_cf[w] = cultural_focus(w, dep_party.keys(), mentions)
        weeks_r_cf[w] = cultural_focus(w, dep_party.keys(), retweets)

    plot_similarity(dep_party.keys(), weeks, weeks_h_cf, dic_color, dir_out + "cult_focus_hastags",
                    'Cultural Focus (1 - Entropy)')
    plot_similarity(dep_party.keys(), weeks, weeks_m_cf, dic_color, dir_out + "cult_focus_mentions",
                    'Cultural Focus (1 - Entropy)')
    plot_similarity(dep_party.keys(), weeks, weeks_r_cf, dic_color, dir_out + "cult_focus_retweets",
                    'Cultural Focus (1 - Entropy)')

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

    plot_reproduction(dep_party.keys(), weeks, party_h_cr, dic_color, dir_out + "cultural_repro_hastags")
    plot_reproduction(dep_party.keys(), weeks, party_m_cr, dic_color, dir_out + "cultural_repro_mentions")
    plot_reproduction(dep_party.keys(), weeks, party_r_cr, dic_color, dir_out + "cultural_repro_retweets")


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

    print("Processando Institutioness e Burstiness")
    h_text  = "\nInstitutionness por partido\n"
    m_text = ''
    r_text = ''
    h_text2 = "\nBurstiness por partido\n"
    m_text2 = ''
    r_text2 = ''
    for party in dep_party.keys():
        h_text += "\n"+party+"\n"
        m_text += "\n"+party + "\n"
        r_text += "\n"+party + "\n"
        h_text2 += "\n"+party + "\n"
        m_text2 += "\n"+party + "\n"
        r_text2 += "\n"+party + "\n"
        h_text += str(sorted(party_h_hi[party].items(), key=lambda x: x[1], reverse=True )[:10]) + "\n"
        m_text += str(sorted(party_m_hi[party].items(), key=lambda x: x[1], reverse=True )[:10]) + "\n"
        r_text += str(sorted(party_r_hi[party].items(), key=lambda x: x[1], reverse=True )[:10]) + "\n"
        bu = party_h_bu[party]
        for x in bu.items():
            if len(x[1]) != 0:
                h_text2 += str(x[1]) + "\n"
        bu = party_m_bu[party]
        for x in bu.items():
            if len(x[1]) != 0:
                m_text2 += str(x[1]) + "\n"
        bu = party_r_bu[party]
        for x in bu.items():
            if len(x[1]) != 0:
                r_text2 += str(x[0])+"\n"+str(x[1]) + "\n"

    t = h_text+"\n"+m_text+"\n"+r_text+"\n"+ h_text2+"\n"+m_text2+"\n"+r_text2+"\n"

    print("Saving file")
    f = open(dir_out + "Burst_institutioness.txt", 'w+')
    f.write(t)

    f.close()
