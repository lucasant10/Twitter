from text_processor import TextProcessor
import json
from read_twitter import ReadTwitter
from collections import Counter
import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx


def days2time(days):
    #1380844800000  = 04/10/2013, 86400000 = 1 day 
    return 1380844800000+(days*86400000)

def days2timeInterval(day1, day2):
    #1380844800000  = 04/10/2013, 86400000 = 1 day 
    return (1380844800000+(day1*86400000)), (1380844800000+(day2*86400000))

if __name__=='__main__':

    dir_in = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/"
    dir_out = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/plot/"
    excel_path = "/Users/lucasso/Dropbox/Twitter_Marcelo/Arquivo Principal da Pesquisa - Quatro Etapas.xls"
    sheet_name = "amostra"
    col = 4
    rt = ReadTwitter(dir_in, excel_path, sheet_name, col )
    tp = TextProcessor()
   

    id_rep, names = rt.names_from_xls()
    
    for idx in range(len(names)):

        tweets = list()
        graphs = list()
        tw = nx.Graph()
        data = rt.tweets_election_data(id_rep[idx])
        
        diction = {k:v for (k,v) in data.items()}

        for i in diction:
            tweets.append(list(itertools.chain.from_iterable(tp.text_process(diction[i].split()))))

        for tweet in tweets:
            for u,v in itertools.combinations(tweet,2):
                if tw.has_edge(u,v):
                    tw[u][v]['weight'] += 1
                else:
                    tw.add_edge(u, v, weight=1)
                
        nx.write_gml(tw , dir_out+names[idx]+".gml")
    
    files = list()
    for file in os.listdir(dir_out):
        if file.endswith(".gml"):
            files.append(file)

    comb_graph = nx.Graph()
    for f_graph in files:
        c =  nx.read_gml(dir_out+f_graph)
        for u,v,data in c.edges_iter(data=True):
            w = data['weight']
            if comb_graph.has_edge(u,v):
                comb_graph[u][v]['weight'] += w
            else:
                comb_graph.add_edge(u, v, weight=w)
    nx.write_gml(comb_graph , dir_out+"combined.gml")
