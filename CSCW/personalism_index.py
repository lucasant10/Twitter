import sys
sys.path.append('../')
import configparser
import topic_BTM as btm
import numpy as np
from collections import Counter, defaultdict
from assign_topics import AssingTopics
import pymongo
import math
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY


def date_tw(time):
    return datetime.fromtimestamp((time / 1000)).strftime('%m/%y')


def get_dates():
    first = datetime.fromtimestamp(1380596400)
    time = datetime.fromtimestamp(1443668300)
    dates = list(rrule(MONTHLY, dtstart=first, until=time))
    dates = {x.strftime('%m/%y'): 0 for x in dates}
    dates.update({'02/15': 0, '02/14': 0})
    return dates


def vocab(path):
    voca = btm.read_voca(dir_btm + path)
    inv_voca = {v: k for k, v in voca.items()}
    return voca, inv_voca


def processing_topic(pz, inv_voca, voca, zw_pt, top_k):
    topics = []
    vectors = np.zeros((len(pz), len(inv_voca)))
    for k, l in enumerate(open(zw_pt)):
        vs = [float(v) for v in l.split()]
        vectors[k, :] = vs
        wvs = zip(range(len(vs)), vs)
        wvs = sorted(wvs, key=lambda d: d[1], reverse=True)
        topics.append([voca[w] for w, v in wvs[:top_k]])
        k += 1
    return topics


def jaccard_distance(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)


def jaccard_topics(t_base, topics):
    tmp = list()
    for topic in topics:
        tmp.append(jaccard_distance(t_base, topic))
    return (tmp.index(max(tmp)), max(tmp))


def list2text(topics):
    txt = ''
    for k, topic in enumerate(topics):
        txt += "%d :" % (k + 1) + ' '.join(topic) + '\n'
    return txt


def load_file(dir_in, file_name):
    text = list()
    tweets = open(dir_in + file_name, "r")
    for l in tweets:
        text.append(l.split())
    return text


def plotly_dist(person_cond, pol_cond):
    map_color1 = {
        'reeleitos': 'rgb(128,0,128)', 'nao_eleitos': 'rgb(255, 128, 0)', 'novos': 'rgb(0,100,80)'}
    map_l = {'novos': 'N', 'reeleitos': 'R', 'nao_eleitos': 'L'}
    data = list()
    for condition, counter in person_cond.items():
        print(condition)
        labels, person_values = zip(
            *sorted(counter.items(), key=lambda i: i[0].split('/')[::-1]))
        _, pol_values = zip(
            *sorted(pol_cond[condition].items(), key=lambda i: i[0].split('/')[::-1]))
        data.append(go.Scatter(
            y=person_values,
            x=labels,
            mode="lines",
            line=go.Line(color=map_color1[condition], dash='line', width=2),
            name="personalism - %s" % map_l[condition],
            opacity=0.8))
        data.append(go.Scatter(
            y=pol_values,
            x=labels,
            mode="lines",
            line=go.Line(color=map_color1[condition], dash='dashdot', width=2),
            name="political - %s" % map_l[condition],
            opacity=0.8))

        data.append(go.Scatter(
            x=['09/14', '09/14'],
            y=[0, 14000],
            mode="lines",
            line=go.Line(color="grey", width=2),
            showlegend=False
        )
        )

        layout = go.Layout(
            xaxis=dict(
                title='Months from 10/01/2013 to 10/01/2015',
                nticks=24,
                titlefont=dict(
                    family='Arial, sans-serif',
                    color='grey'
                )
            ),
            yaxis=dict(
                title='# of Tweets',
                titlefont=dict(
                    family='Arial, sans-serif',
                    color='grey'
                )
            )
        )
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='distribution')


def plotly_popularity(pop_cond):
    map_color1 = {
        'reeleitos': 'rgb(128,0,128)', 'nao_eleitos': 'rgb(255, 128, 0)', 'novos': 'rgb(0,100,80)'}
    map_l = {'novos': 'N', 'reeleitos': 'R', 'nao_eleitos': 'L'}
    data = list()
    fig = tools.make_subplots(
        rows=1, cols=3, shared_yaxes=True, specs=[[{}, {}, {}]])
    col = 1
    for condition, counter in pop_cond.items():
        topics, qtd_tw = zip(*counter.items())
        fig.append_trace(go.Bar(
            y=topics,
            x=qtd_tw,
            marker=dict(color=map_color1[condition]),
            name="personalism - %s" % map_l[condition],
            opacity=0.8,
            orientation='h'),
            1, col)
        col += 1

    fig['layout'].update(
        xaxis=dict(
            title='# of tweets',
            nticks=20,
            domain=[0, 1], 
            autorange =  True, 
            titlefont=dict(
                family='Arial, sans-serif',
                color='grey'
            )
        ),
        yaxis=dict(
            title='topic index',
            autotick=False,
            titlefont=dict(
                family='Arial, sans-serif',
                color='grey'
            )
        )
    )
    plot(fig, filename='popularity_dist')


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

    cond_tw = defaultdict(list)
    for tw in tweets:
        cond_tw[tw['cond_55']].append((tw['text_processed'], date_tw(
            int(tw['created_at'])), (int(tw['retweets']) + int(tw['favorites']))))

    # cond_index = dict()
    # for cond, tws in cond_tw.items():
    #     tweets = [tw[0].split() for tw in tws]
    #     total, target = at.adjetive_index(tweets, at.topics, person_topicos)
    #     cond_index[cond] = "%0.2f" % (sum(target.values()) / sum(total.values()))

    # print(cond_index)

    person_cond = dict()
    pol_cond = dict()
    pop_cond = defaultdict(dict)
    count = defaultdict(int)
    for cond, tws in cond_tw.items():
        for tw in tws:
            tweet = tw[0].split()
            date = tw[1]
            popularity = tw[2]
            index = at.get_text_topic(tweet, at.topics)
            # topic popularity
            count[index] += popularity
            if cond in pop_cond:
                pop_cond[cond][index] += popularity
            else:
                pop_cond[cond] = {x: 0 for x in range(0, 20)}
                pop_cond[cond][index] += popularity

            # personalism index
            if index in person_topicos:
                if cond in person_cond:
                    person_cond[cond][date] += 1
                else:
                    person_cond[cond] = get_dates()
                    person_cond[cond][date] += 1
            else:
                if cond in pol_cond:
                    pol_cond[cond][date] += 1
                else:
                    pol_cond[cond] = get_dates()
                    pol_cond[cond][date] += 1
    plotly_dist(person_cond, pol_cond)

    pop_cont_norm = defaultdict(dict)
    for cond, dic in pop_cond.items():
        tmp = dict()
        for k in dic:
            z = (dic[k] / count[k]) if count[k] != 0 else 0
            tmp[k] = z
        pop_cont_norm[cond] = tmp

    plotly_popularity(pop_cont_norm)
