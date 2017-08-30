import sys

sys.path.append('../')
import configparser
import pickle
import pandas as pd
import datetime
import numpy as np
import TVGL as tvgl

def tweet_day(date):
    dt_tw = pd.to_datetime(date * 1000000)
    return (dt_tw - datetime.datetime(2013, 10, 4)).days

def is_election(date):
    dt_tw = pd.to_datetime(date * 1000000)
    return (datetime.datetime(2013, 10, 4) <= dt_tw <= datetime.datetime(2015, 10, 4))

cf = configparser.ConfigParser()
cf.read("../file_path.properties")
path = dict(cf.items("file_path"))
dir_out = path['dir_out']

with open(dir_out + "time_doc_parl_2.pck", 'rb') as handle:
    parl = pickle.load(handle)

tw_parl = list()
for p in parl:
    days = np.zeros(730)
    for tw in p:
        if (is_election(tw[0])):
            days[tweet_day(tw[0])] += 1
    tw_parl.append(days)

with open(dir_out + "tw_per_day_parl.pck", 'wb') as handle:
    pickle.dump(tw_parl, handle)

# transpose = np.array(tw_parl).T
#
# lamb = 2.5
# beta = 12
# lengthOfSlice = 7
# thetaSet = tvgl.TVGL(transpose, lengthOfSlice, lamb, beta, indexOfPenalty = 3, verbose=True)
# print(thetaSet)
np.linalg.norm(thetaSet[2]-thetaSet[1],'fro')












