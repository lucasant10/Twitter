import sys
import os
sys.path.append('../')
import argparse
import configparser
import numpy as np
from matplotlib import pyplot as plt
import pymongo
from political_classification import PoliticalClassification
import math
import random
from collections import Counter, defaultdict
import matplotlib.patches as mpatches
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY
from inactive_users import Inactive_Users
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def date_tw(time):
    return datetime.fromtimestamp((time / 1000)).strftime('%m/%y')


def get_dates():
    first = datetime.fromtimestamp(1380596400)
    time = datetime.fromtimestamp(1443668300)
    dates = list(rrule(MONTHLY, dtstart=first, until=time))
    dates = {x.strftime('%m/%y'): 0 for x in dates}
    dates.update({'02/15': 0, '02/14': 0})
    return dates

def plotly_dist(p_condition, np_condition):
    map_color1 = {
        'reeleitos': 'rgb(128,0,128)', 'nao_eleitos': 'rgb(255, 128, 0)', 'novos': 'rgb(0,100,80)'}
    map_l = {'novos': 'NC', 'reeleitos': 'RE', 'nao_eleitos': 'LS'}
    data = list()
    for condition, counter in p_condition.items():
        print(condition)
        labels, p_values = zip(
            *sorted(counter.items(), key=lambda i: i[0].split('/')[::-1]))
        _, np_values = zip(
            *sorted(np_condition[condition].items(), key=lambda i: i[0].split('/')[::-1]))
        data.append(go.Scatter(
            y=p_values,
            x=labels,
            mode="lines",
            line=go.Line(color=map_color1[condition], dash='line', width=2),
            name="political - %s" % map_l[condition],
            opacity=0.8))
        data.append(go.Scatter(
            y=np_values,
            x=labels,
            mode="lines",
            line=go.Line(color=map_color1[condition], dash='dashdot', width=2),
            name="non-political - %s" % map_l[condition],
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
            legend=dict(
                x=0,
                y=1,
                traceorder='normal',
                font=dict(
                    family='sans-serif',
                    size=8,
                    color='#000'
                )
            ),
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
                nticks=12,
                titlefont=dict(
                    family='Arial, sans-serif',
                    color='grey'
                )
            )
        )
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='time_distribution')
if __name__ == "__main__":
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_btm = path['dir_btm']
    dir_in = path['dir_in']

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb

    tweets = db.tweets.find({'created_at': {'$gte': 1380596400000, '$lt': 1443668300000}, 'cond_55': {'$exists': True} })

    pc = PoliticalClassification('model_cnn.h5', 'dict_cnn.npy', 18)
    
    inactive = Inactive_Users()
    inact_users = inactive.inactive_users()

    p_condition = dict()
    n_p_condition = dict()

    print('processing tweets ...')
    print(get_dates())
    for tweet in tweets:
        if tweet['user_id'] in inact_users:
            continue
        month = date_tw(int(tweet['created_at']))
        cond = tweet['cond_55']
        if pc.is_political(tweet['text_processed']):
            if cond not in p_condition:
                p_condition[cond] = get_dates()
            p_condition[cond][month] += 1
        else:
            if cond not in n_p_condition:
                n_p_condition[cond] = get_dates()
            n_p_condition[cond][month] += 1

    print('plotting distribution ...')
    plotly_dist(p_condition, n_p_condition)
