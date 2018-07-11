import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('../')
import configparser
import numpy as np
from matplotlib import pyplot as plt
import pymongo
from political_classification import PoliticalClassification
import math
import random
from collections import defaultdict
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from scipy import stats
import os
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

def plotly_dist(political, non_political, filename):
    map_c = {'novos': 'green', 'reeleitos': 'purple', 'nao_eleitos': 'orange'}
    map_l = {'novos': 'NC', 'reeleitos': 'RE', 'nao_eleitos': 'LS'}
    data = list()
    for condition, counter in political.items():
        labels, p_influence = zip(
            *sorted(counter.items(), key=lambda i: i[0].split('/')[::-1]))
        _, np_influence = zip(
            *sorted(non_political[condition].items(), key=lambda i: i[0].split('/')[::-1]))
        data.append(go.Scatter(
            y=p_influence,
            x=labels,
            mode="lines",
            line=go.Line(color=map_c[condition], dash='line', width=2),
            name="political - %s" % map_l[condition],
            opacity=0.8))
        data.append(go.Scatter(
            y=np_influence,
            x=labels,
            mode="lines",
            line=go.Line(color=map_c[condition], dash='dashdot', width=2),
            name="non-political - %s" % map_l[condition],
            opacity=0.8))

        data.append(go.Scatter(
            x=['10/14', '10/14'],
            y=[0, (max(max(p_influence), max(np_influence)) + 10000)],
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
                title='%s' % filename,
                titlefont=dict(
                    family='Arial, sans-serif',
                    color='grey'
                )
            )
        )
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename=filename)

def plot_pdf(politics, non_politics, label):
    map_l = {'novos': 'New', 'reeleitos': 'Reelected', 'nao_eleitos': 'Loser'}
    num_bins = 100
    for cond, values in politics.items():
        p_values = np.where(np.isneginf(np.log(values)), 0, np.log(values))
        np_values = np.where(np.isneginf(np.log(non_politics[cond])), 0, np.log(non_politics[cond]))
        # p_counts, p_bin = np.histogram (values, bins=num_bins, normed=True)
        # np_counts, np_bin = np.histogram (non_politics[cond], bins=num_bins, normed=True)
        figure = plt.figure(figsize=(15, 8))
        ax = figure.add_subplot(111)
        ax.hist (p_values, num_bins, label= 'political', normed=1)
        ax.hist (np_values, num_bins, label= 'non-political', normed=1)
        ax.legend(loc='upper left')
        ax.set_xlabel('%s' % label)
        ax.set_ylabel('% of tweets')
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        #ax.set_yticks(np.arange(0.5, 1.1, step=0.1))
        #ax.set_xticks(np.arange(0, 1, step=1))
        ax.set_title("%s %s-PDF " %(map_l[cond], label))
        plt.savefig(dir_in + 'pdf_%s_%s.png' % (label, map_l[cond]))
        plt.clf()

def plot_cdf(politics, non_politics, label):
    map_l = {'novos': 'New', 'reeleitos': 'Reelected', 'nao_eleitos': 'Loser'}
    num_bins = 100
    p_values = list()
    np_values = list()
    for cond, values in politics.items():
        p_values += values
        np_values += non_politics[cond]
    print(stats.ks_2samp(p_values,np_values))
    # p_values = np.where(np.isneginf(np.log10(p_values)), 0, np.log10(p_values))
    # np_values = np.where(np.isneginf(np.log10(np_values)), 0, np.log10(np_values))
    p_counts, p_bin_edges = np.histogram (p_values, bins=num_bins, normed=True)
    np_counts, np_bin_edges = np.histogram (np_values, bins=num_bins, normed=True)
    p_cdf = np.cumsum (p_counts)
    np_cdf = np.cumsum (np_counts)
    figure = plt.figure(figsize=(15, 8))
    ax = figure.add_subplot(111)
    ax.plot (p_bin_edges[1:], p_cdf/p_cdf[-1], label= 'political')
    ax.plot (np_bin_edges[1:], np_cdf/np_cdf[-1], label='non-political')
    ax.legend(loc='upper left')
    ax.set_xlabel('# of %s tweets'% str.lower(label))
    ax.set_ylabel('F(x)')
    ax.xaxis.label.set_size(28)
    ax.yaxis.label.set_size(28)  
    #ax.set_xscale("log", basex=2)
    #ax.set_yticks(np.arange(0.5, 1.0, step=0.1))
    ax.set_xticks(np.arange(0, 50, step=1))
    plt.xlim(xmax=50)
    plt.xlim(xmin=0)  
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.savefig(dir_in + 'cdf_%s.png' % (label))
    plt.clf()


if __name__ == "__main__":
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']

    pc = PoliticalClassification('cnn_s300.h5', 'cnn_s300.npy', 18)

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb
    tweets = db.tweets.find({'created_at': {
                            '$gte': 1380596400000, '$lt': 1443668400000}, 'cond_55': {'$exists': True}})

    inactive = Inactive_Users()
    inact_users = inactive.inactive_users()

    pop_score_politics = dict(
        {'nao_eleitos': get_dates(), 'reeleitos': get_dates(), 'novos': get_dates()})
    pop_score_non_politics = dict(
        {'nao_eleitos': get_dates(), 'reeleitos': get_dates(), 'novos': get_dates()})
    
    f_dist_politics = dict(
        {'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})
    f_dist_non_politics = dict(
        {'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})
    rt_dist_politics = dict(
        {'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})
    rt_dist_non_politics = dict(
        {'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})

    pop_score_dist_politics = dict(
        {'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})
    pop_score_dist_non_politics = dict(
        {'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})

    print('processing tweets')
    for tweet in tweets:
        if tweet['user_id'] in inact_users:
            continue
        favorites = int(tweet['favorites'])
        retweets = int(tweet['retweets'])
        if pc.is_political(tweet['text_processed']):
            pop_score_politics[tweet['cond_55']][date_tw(tweet['created_at'])] += (retweets +favorites)
            
            f_dist_politics[tweet['cond_55']].append(favorites)
            rt_dist_politics[tweet['cond_55']].append(retweets)

            pop_score_dist_politics[tweet['cond_55']].append((retweets + favorites))
        else:
            pop_score_non_politics[tweet['cond_55']][date_tw(tweet['created_at'])] += (retweets +favorites)

            f_dist_non_politics[tweet['cond_55']].append(favorites)
            rt_dist_non_politics[tweet['cond_55']].append(retweets)

            pop_score_dist_non_politics[tweet['cond_55']].append((retweets + favorites))
    
    # plotly_dist(f_politics, f_non_politics, 'favorites')
    # plotly_dist(rt_politics, rt_non_politics, 'retweets')

    # plot_cdf(f_dist_politics, f_dist_non_politics, 'Favorites')
    # plot_cdf(rt_dist_politics, rt_dist_non_politics, 'Retweets')

    plot_cdf(f_dist_politics, f_dist_non_politics, 'Favorites')
    plot_cdf(rt_dist_politics, rt_dist_non_politics, 'Retweets')

    plot_cdf(pop_score_dist_politics, pop_score_dist_non_politics, 'Pop-score')
    plotly_dist(pop_score_politics, pop_score_non_politics, 'Pop-score')

