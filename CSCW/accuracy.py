import sys
import os
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
from political_classification import PoliticalClassification
from sklearn.metrics import classification_report, precision_recall_fscore_support
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def date_tw(time):
    return datetime.fromtimestamp((time / 1000)).strftime('%m/%y')

def plotly_acc(dist1, dist2):
    data = list()
    data.append(go.Box(y=dist1, name="F1 per congressmen"))
    data.append(go.Box(y=dist2, name="F1 per month"))
    layout = go.Layout( 
        autosize=False,
        # width=500,
        # height=500,
        # #showlegend=False,
        font=dict( size=24 )
        )
    fig = go.Figure(data = data, layout=layout)
    plot(fig, filename = "accuracy" )


if __name__ == "__main__":
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    pc = PoliticalClassification('cnn_s300.h5', 'cnn_s300.npy', 18)

    tweets = db.tweets.find({'created_at': {
                            '$gte': 1380596400000, '$lt': 1443668400000}, 'cond_55': {'$exists': True}})

    tweet_pol = defaultdict(list)
    tweet_month = defaultdict(list)

    for tw in tweets:
        tweet_pol[tw["user_id"]].append(tw["text_processed"])
        tweet_month[date_tw(tw["created_at"])].append(tw["text_processed"])

    tweet_pol = {k:v for k,v in tweet_pol.items()  if len(v)>20}
    tweet_month = {k:v for k,v in tweet_month.items()  if len(v)>20}

    dep_l = list()
    for dep, texts in tweet_pol.items():
        y_pred = list()
        y_real = list()
        for tx in texts:
            y_real.append(1)
            if pc.is_political(tx):
                y_pred.append(1)
            else:
                y_pred.append(0)
        dep_l.append(precision_recall_fscore_support(y_real, y_pred, average='weighted')[2])

    month_l = list()
    for dep, texts in tweet_month.items():
        y_pred = list()
        y_real = list()
        for tx in texts:
            y_real.append(1)
            if pc.is_political(tx):
                y_pred.append(1)
            else:
                y_pred.append(0)
        month_l.append(precision_recall_fscore_support(y_real, y_pred, average='weighted')[2])

    plotly_acc(dep_l, month_l)