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
import random
from collections import defaultdict
from beautifultable import BeautifulTable
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import scipy.stats as st
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import seaborn as sns
import os
from inactive_users import Inactive_Users
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def safe_div(x,y):
    if y==0: return 0
    return x/y

def plot_tsne(dep_vector, filename):
    map_l = {'novos': 'Elected', 'reeleitos': 'Reelected',
             'nao_eleitos': 'Not Elected'}
    map_c = {'novos': 'green', 'reeleitos': 'purple',
             'nao_eleitos': 'orange'}
    tsne = TSNE(n_components=2, random_state=0)
    figure = plt.figure(figsize=(15, 8))
    ax = figure.add_subplot(111)
    colors = [map_c[x[-1]] for x in dep_vector]
    vec = [x[:-1] for x in dep_vector]
    vec = np.where(np.isneginf(np.log2(vec)), 0, np.log2(vec))
    Y = tsne.fit_transform(vec)
    ax.scatter(Y[:, 0], Y[:, 1], c=colors)
    ax.set_title("TSNE Non Political by periods")
    plt.savefig(dir_in + filename)


def plot_pca(dep_vector, filename):
    map_l = {'novos': 'Elected', 'reeleitos': 'Reelected',
             'nao_eleitos': 'Not Elected'}
    map_c = {'novos': 'green', 'reeleitos': 'purple',
             'nao_eleitos': 'orange'}
    pca = PCA(n_components=2)
    figure = plt.figure(figsize=(15, 8))
    ax = figure.add_subplot(111)
    colors = [map_c[x[-1]] for x in dep_vector]
    vec = [x[:-1] for x in dep_vector]
    vec = np.where(np.isneginf(np.log2(vec)), 0, np.log2(vec))
    Y = pca.fit_transform(vec)
    exp = pca.explained_variance_ratio_
    print(pca.explained_variance_ratio_)
    ax.scatter(Y[:, 0], Y[:, 1], c=colors)
    #ax.set_title("PCA Non Political by periods")
    plt.savefig(dir_in + filename)
    return Y


def kde_scipy( vals1, vals2, limits, N ):

    #vals1, vals2 are the values of two variables (columns)
    #(a,b) interval for vals1; usually larger than (np.min(vals1), np.max(vals1))
    #(c,d) -"-          vals2 

    x=np.linspace(limits[0], limits[1] ,N)
    y=np.linspace(limits[2], limits[3] ,N)
    X,Y=np.meshgrid(x,y)
    positions = np.vstack([Y.ravel(), X.ravel()])

    values = np.vstack([vals1, vals2])
    kernel = st.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    return [x, y, Z]

def make_kdeplot(varX, varY,limits, N, colorsc, title):
    #varX, varY are lists, 1d numpy.array(s), or dataframe columns, storing the values of two variables
    x, y, Z = kde_scipy(varY, varX, limits, N )

    data = go.Data([
       go.Contour(
           z=Z,
           x=x,
           y=y,
           colorscale=colorsc,
           #reversescale=True,
           opacity=0.9,
           contours=go.Contours(
               showlines=False)
        ),
     ])

    layout = go.Layout(
        title= title,
        font= go.Font(family='Georgia, serif',  color='#635F5D'),
        showlegend=False,
        autosize=False,
        width=650,
        height=650,
        xaxis=go.XAxis(
            range=[limits[0],limits[1]],
            showgrid=False,
            nticks=10
        ),
        yaxis=go.YAxis(
            range=[limits[2],limits[3]],
            showgrid=False,
            nticks=10
        ),
        margin=go.Margin(
            l=40,
            r=40,
            b=85,
            t=100,
        ),
    )
    fig = go.Figure(data = data, layout= layout)
    plot(fig, filename = 'kde_ratio')

def plot_kde(Y, filename, colors, color, lista, cond):
    plt.clf()
    sns.set(style="white")
    sns.kdeplot(Y[:, 0], Y[:, 1], shade=True, bw=.15, cmap=colors, shade_lowest=False, alpha=0.6)
    sns.regplot(x=Y[:, 0], y=Y[:, 1], fit_reg=False, scatter_kws={"color":color,"alpha":0.7} )
    plt.ylim(-4, 6)
    plt.xlim(-2, 6)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    for i, v in enumerate(lista):
        plt.annotate('%s%d'%(cond, (i+1)), xy=(v[0], v[1]), textcoords='offset points', fontsize=25)
    plt.savefig(dir_in + filename)

def plot_ratio(ratio, filename, lista, person):
    label = ['10/13','03/14', '07/14', '10/14', '12/14', '04/15']
    items = ratio[lista]
    plt.clf()
    sns.set(style="white")
    for i, v in enumerate(items):
        plt.plot(v, linestyle='-', marker='o')
    plt.legend(['%s1'% person, '%s2'% person, '%s3'% person], fontsize=25)
    plt.xticks(range(6), label,fontsize=35)
    plt.yticks(fontsize=35)
    plt.savefig(dir_in + filename)

def plot_values(total, filename, lista, person):
    label = ['10/13','03/14', '07/14', '10/14', '12/14', '04/15']
    items = total[lista]
    plt.clf()
    sns.set(style="white")
    for i, v in enumerate(items):
        plt.plot(v, linestyle='-', marker='o')
    plt.legend(['%s1'% person, '%s2'% person, '%s3'% person], fontsize=25)
    plt.xticks(range(6), label,fontsize=35)
    plt.yticks(fontsize=35)
    #plt.yscale('log', basey=2)
    plt.savefig(dir_in + filename)


def deputy_index(Y, lista):
    x = np.around(Y[:,0], decimals=2)
    y = np.around(Y[:,1], decimals=2)
    idx = list()
    for i in lista:
        idx.append(np.argwhere((x==i[0]) & (y==i[1])))
    return idx

def print_table(table, Y):
    t_print =list()
    for i, v in enumerate(table):
        tmp = list()
        tmp += v
        tmp += [Y[i,0],Y[i,1]]
        t_print.append(tmp)
    with open(dir_in+'table_pedro.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(t_print)

if __name__ == "__main__":
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_dbx']

    # pre election
    p1 = (1380596400000, 1393642800000)
    p2 = (1393642800000, 1404183600000)
    p3 = (1404183600000, 1412132400000)
    # post election
    p4 = (1412132400000, 1417402800000)
    p5 = (1417402800000, 1427857200000)
    p6 = (1427857200000, 1443668300000)
    cubehelix_cs=[[0.0, '#fcf9f7'],
            [0.16666666666666666, '#edcfc9'],
            [0.3333333333333333, '#daa2ac'],
            [0.5, '#bc7897'],
            [0.6666666666666666, '#925684'],
            [0.8333333333333333, '#5f3868'],
            [1.0, '#2d1e3e']]

    pc = PoliticalClassification('cnn_s300.h5', 'cnn_s300.npy', 18)

    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb

    inactive = Inactive_Users()
    inact_users = inactive.inactive_users()

    periods = [p1, p2, p3, p4, p5, p6]
    politics = dict(
        {'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    non_politics = dict(
        {'nao_eleitos': dict(), 'reeleitos': dict(), 'novos': dict()})
    count_tweets = defaultdict(int)
    for period in periods:
        print('getting data')
        tweets = db.tweets.find({'created_at': {'$gte': period[0], '$lt': period[1]},
                                 'cond_55': {'$exists': True}})
        #tweets = db.tweets.aggregate([ { '$sample': { 'size': 10000 }},
        #    {'$match': {'created_at': {'$gte': period[0], '$lt': period[1]},
        #        'cond_55': {'$exists': True}}}], allowDiskUse=True)

        print('processing tweets')
        for tweet in tweets:
            if tweet['user_id'] in inact_users:
                continue
            count_tweets[tweet['user_id']] += 1
            if pc.is_political(tweet['text_processed']):
                if tweet['user_id'] not in politics[tweet['cond_55']]:
                    politics[tweet['cond_55']
                             ][tweet['user_id']] = defaultdict(int)
                politics[tweet['cond_55']][tweet['user_id']][period[0]] += 1
            else:
                if tweet['user_id'] not in non_politics[tweet['cond_55']]:
                    non_politics[tweet['cond_55']
                                 ][tweet['user_id']] = defaultdict(int)
                non_politics[tweet['cond_55']
                             ][tweet['user_id']][period[0]] += 1

    print('creating vector table')
    table = ''
    vec_n_pol = list()
    vec_pol = list()
    for cond, dep in non_politics.items():
        for d, val in dep.items():
            table += '%d,%d,%d,%d,%d,%d,%d,%s\n' % (
                val[p1[0]], val[p2[0]], val[p3[0]], val[p4[0]], val[p5[0]], val[p6[0]], count_tweets[d], cond)
            vec_n_pol.append([val[p1[0]], val[p2[0]], val[p3[0]],
                              val[p4[0]], val[p5[0]], val[p6[0]], count_tweets[d], cond])
            if(d in politics[cond]):
                vec_pol.append([politics[cond][d][p1[0]], politics[cond][d][p2[0]], politics[cond][d][p3[0]],
                                politics[cond][d][p4[0]], politics[cond][d][p5[0]], politics[cond][d][p6[0]], count_tweets[d], cond])
            else:
                vec_pol.append([0,0,0,0,0,0, count_tweets[d], cond])
            
    # both = list()
    # for i, t in enumerate(vec_n_pol):
    #     tmp = list()
    #     for k, v in enumerate(t[:6]):
    #         tmp += [vec_pol[i][k], v]
    #     tmp += t[6:]
    #     both.append(tmp)

    # with open(dir_in + 'tw_topics/table_np_dep.txt', 'w') as f:
    #     f.write(table)
    # f.close()
    #plot_pca(vec_n_pol, 'pca_log_n_political.png')
    #plot_tsne(vec_n_pol, 'tsne_log_n_political.png')

    ratio = list()
    table = list()
    for i, t in enumerate(vec_pol):
        tmp = list()
        tmp2 = list()
        for k, v in enumerate(t[:6]):

            tmp += [safe_div(v, (vec_n_pol[i][k] + v))]
            tmp2 += [(vec_n_pol[i][k] + v )]
        tmp += t[7:]
        tmp2 += tmp
        ratio.append(tmp)
        table.append(tmp2)
    
    Y = plot_pca(ratio, 'pca_log_political_ratio.png')
    print_table(table,Y)
    # N=200
    # limits =[-6,11,-4,6]
    # fig=make_kdeplot(Y[:, 0], Y[:, 1], limits,
    #              N, cubehelix_cs,'kde plot of two sets of data' )
    r = list()
    l =list()
    n = list()
    for i, v in enumerate(ratio):
        if v[-1] == 'reeleitos':
            r.append(Y[i])
        elif v[-1] == 'novos':
            n.append(Y[i])
        else:
            l.append(Y[i])

    r = np.asarray(r)
    n = np.asarray(n)
    l = np.asarray(l)

    dep_l = [[-1.31, 0.77],[ 0.  ,-0.05], [ 2.75,-2.2 ]]
    dep_r = [[-1.23, 0.89], [-0.12,-0.01],[ 4.72,-0.97]]
    dep_n = [[-1.48, 0.65], [ 0.02,-0.02],[ 3.84,-2.6 ]]
    
    plot_kde(np.asarray(r), 'kde_ratio_sns_reelected.png', "Purples",'purple', dep_r,'RE')
    plot_kde(np.asarray(n), 'kde_ratio_sns_new.png', "Greens",'green', dep_n, 'NC')
    plot_kde(np.asarray(l), 'kde_ratio_sns_loser.png', "Oranges", 'orange', dep_l, 'LS')

    index_l = deputy_index(l, dep_l)
    index_r = deputy_index(r, dep_r)
    index_n = deputy_index(n, dep_n)

    print(index_l)
    print(index_r)
    print(index_n)

    index_l = [188,18,88]
    index_r = [231, 87, 100]
    index_n = [144, 16, 43]

    r = list()
    l =list()
    n = list()
    for i, v in enumerate(ratio):
        if v[-1] == 'reeleitos':
            r.append(v[:-1])
        elif v[-1] == 'novos':
            n.append(v[:-1])
        else:
            l.append(v[:-1])

    r = np.asarray(r)
    n = np.asarray(n)
    l = np.asarray(l)
    
    plot_ratio(l,'dep_ratio_sns_loser.png', index_l ,'LS')
    plot_ratio(r,'dep_ratio_sns_reelected.png', index_r,'RE')
    plot_ratio(n,'dep_ratio_sns_new.png', index_n,'NC')

    total = list()
    for i, t in enumerate(vec_pol):
        tmp = list()
        for k, v in enumerate(t[:6]):
            tmp += [(v + vec_n_pol[i][k])]
        tmp += t[7:]
        total.append(tmp)

    r = list()
    l =list()
    n = list()
    for i, v in enumerate(total):
        if v[-1] == 'reeleitos':
            r.append(v[:-1])
        elif v[-1] == 'novos':
            n.append(v[:-1])
        else:
            l.append(v[:-1])

    r = np.asarray(r)
    n = np.asarray(n)
    l = np.asarray(l)

    plot_values(l,'dep_total_sns_loser.png', index_l,'LS' )
    plot_values(r,'dep_total_sns_reelected.png', index_r, 'RE')
    plot_values(n,'dep_total_sns_new.png', index_n,'NC')
    


    # f =  open(dir_in + "CSCW/pontos.txt", 'w')
    # f.write(np.array2string(l, precision=2, separator=',',suppress_small=True))
    # f.write('\n')
    # f.write(np.array2string(r, precision=2, separator=',',suppress_small=True))
    # f.write('\n')
    # f.write(np.array2string(n, precision=2, separator=',',suppress_small=True))
    # f.close()

    
    
    # f_pol = dict({'nao_eleitos': list(), 'reeleitos': list(), 'novos': list()})
    # for cond, dep in non_politics.items():
    #     for _, val in dep.items():
    #         vec = [val[p1[0]], val[p2[0]], val[p3[0]], val[p4[0]]]
    #         vec = vec / np.sum(vec)
    #         f_pol[cond].append(vec)

    # plot_tsne(f_pol, 'tsne_non_political.png')
    # plot_pca(f_pol, 'pca_non_political.png')
