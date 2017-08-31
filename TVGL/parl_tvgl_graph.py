import sys

sys.path.append('../')
import configparser
import pickle
import pandas as pd
import datetime
import numpy as np
import TVGL as tvgl
import networkx as nx
from matplotlib import pyplot as plt

cf = configparser.ConfigParser()
cf.read("../file_path.properties")
path = dict(cf.items("file_path"))
dir_out = path['dir_out']

print("processing image")
with open(dir_out + "theta_list.pck", 'rb') as handle:
    theta_list = pickle.load(handle)


t_series= list()
for i in range(len(theta_list)-1):
    tmp = np.linalg.norm(theta_list[i+1]-theta_list[i])
    t_series.append(tmp)

plt.clf()
plt.xticks(range(0,120,5))
plt.plot(range(len(t_series)), t_series,'-o')
plt.show()

#graph = nx.Graph()

plt.clf()
graph = nx.from_numpy_matrix(theta_list[52])
nx.draw(graph)
plt.show()

nx.write_gml(graph,dir_out+"parl_tvgl_election.gml")









