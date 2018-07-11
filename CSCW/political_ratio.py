import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('../')
import argparse
import configparser
import numpy as np
from matplotlib import pyplot as plt
import pymongo
from political_classification import PoliticalClassification
import math
from collections import defaultdict
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def ratio(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return 0


def plot_scatter(political, total, filename):
    map_c = {'reeleitos': 'rgb(128,0,128)',
                               'nao_eleitos': 'rgb(255, 128, 0)', 'novos': 'rgb(0,100,80)'}
    map_l = {'novos': 'N', 'reeleitos': 'R', 'nao_eleitos': 'L'}
    data = list()
    max_val = list()
    pol = list()

    for condition, counter in political.items():
        x = list()
        y = list()
        rt = 0
        for dep, value in counter.items():
            rt = ratio(value, total[dep])
            x.append(value)
            y.append(rt)
            pol.append(rt)
            max_val.append(value)
        data.append(go.Scatter(
            y=y,
            x=x,
            mode="markers",
            marker=dict(
                    size=10,
                    color=map_c[condition],
                    opacity=0.8),
            name="political idx %s" % map_l[condition]))
    data.append(go.Scatter(
        y=[0.5, 0.5],
        x=[0, 10000],
        mode="lines",
        line=go.Line(color="black", width=2),
        showlegend=False))
    data.append(go.Scatter(
        x=[max(max_val),max(max_val)],
        y=[0.75, 0.25],
        mode='text',
        text=['%d%%'%((np.count_nonzero(np.asarray(pol)>0.5)/631*100)), '%d%%'%((np.count_nonzero(np.asarray(pol)<=0.5)/631*100))],
        showlegend=False
        )
    )

    layout=go.Layout(
        width=800,
        height=800,
        font= go.Font(family='Arial, sans-serif', size=20, color='#635F5D'),
        xaxis=dict(
            title='# of tweets',
            nticks=10,
            type='log',
            autorange=True,
            titlefont=dict(
                family='Arial, sans-serif',
                color='grey'
            )
        ),
        yaxis=dict(
            title='political communication index',
            nticks=10,
            titlefont=dict(
                family='Arial, sans-serif',
                color='grey'
            )
        )
    )
    fig=go.Figure(data=data, layout=layout)
    plot(fig, filename=filename)


if __name__ == "__main__":
    cf=configparser.ConfigParser()
    cf.read("../file_path.properties")
    path=dict(cf.items("file_path"))
    dir_in=path['dir_in']

    client=pymongo.MongoClient("mongodb://localhost:27017")
    db=client.twitterdb

    pc=PoliticalClassification('cnn_s300.h5', 'cnn_s300.npy', 18)

    politics=dict(
        {'nao_eleitos': defaultdict(int), 'reeleitos': defaultdict(int), 'novos': defaultdict(int)})
    count_tweets=defaultdict(int)
    print('getting data')
    tweets=db.tweets.find({'created_at': {'$gte': 1380596400000, '$lt': 1443668300000},
                             'cond_55': {'$exists': True}})
    print('processing tweets')
    for tweet in tweets:
        count_tweets[tweet['user_id']] += 1
        if pc.is_political(tweet['text_processed']):
            politics[tweet['cond_55']][tweet['user_id']] += 1

    plot_scatter(politics, count_tweets, 'political_ratio')
