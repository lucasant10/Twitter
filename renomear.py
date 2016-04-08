 #-*- coding: utf-8 -*-
import os
import json
import matplotlib.dates as mdates
from pprint import pprint
from matplotlib import pyplot as plt
from matplotlib.dates import WeekdayLocator
import datetime
import pandas as pd
from pandas.tseries.resample import TimeGrouper
from pandas.tseries.offsets import DateOffset
from matplotlib import dates
import xlrd
from matplotlib import numpy as np
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('utf-8')

class PlotTwitter:

    before =[]
    after=[]
    rep_dic={}
    erro =[]

    def __init__(self, dir_in, dir_out,excel_path, sheet_name, col):
        self.dir_in = dir_in
        self.dir_out = dir_out
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.col = col



    def run(self):

        fnames = ([file for root, dirs, files in os.walk(self.dir_in)
            for file in files if file.endswith('.json')  ])
        self.create_dic(fnames)
        self.save()
        print("No file for these Ids:")
        print(self.erro)
        np.savetxt(self.dir_out+'Ids_not_found.txt', self.erro, delimiter=",", fmt="%s")

    def save(self):

        xls = xlrd.open_workbook(self.excel_path)
        sheet = xls.sheet_by_name(self.sheet_name)

        for i in xrange(sheet.nrows):
            #id_rep = str(int(sheet.cell_value(rowx= i, colx=self.col)))
            id_rep = str(sheet.cell_value(rowx= i, colx=self.col))
            user_name=""
            date_fq =[]
            if (self.rep_dic.has_key(id_rep)):
                #with open(self.dir_in+self.rep_dic[id_rep]) as data_file:
                    os.rename(self.dir_in+self.rep_dic[id_rep], self.dir_in+sheet.cell_value(rowx= i, colx=4)+"_"+self.rep_dic[id_rep])
            else:
                self.erro.append(id_rep)


    def create_dic(self, fnames):
        for fname in fnames:
            self.rep_dic[fname.split('.json',1)[0]] = fname




    def save_before(self,id_rep, user_name, date_fq):
        self.id_rep = id_rep
        self.user_name = user_name
        self.date_fq = date_fq
        before = [date_fq[i] for i in xrange(0,len(date_fq)) if date_fq[i] >= (datetime.datetime(2013,10,04)) and date_fq[i] <= (datetime.datetime(2014,10,04))]
        if len(before) == 0: before.append(datetime.datetime(2100,01,01))
        ones = [1]*len(before)
        idx = pd.DatetimeIndex(before)
        serie = pd.Series(ones, index=idx)
        per_week = serie.resample('W', how='count').fillna(0)
        per_week.plot(color="blue")
        #per_week.plot( marker="o", color="green")
        plt.title(user_name+" - A Year Before Election")
        plt.ylabel("Number of Posts")
        plt.xlabel("Date (Week)")
        plt.ylim(0,150)
        plt.grid(True, which='both')
        plt.xlim(datetime.datetime(2013,10,04), datetime.datetime(2014,10,4))
        plt.savefig(self.dir_out+self.id_rep+"_"+self.user_name+"_before"+".png")
        plt.clf()

    def save_after(self, id_rep, user_name, data):
        self.id_rep = id_rep
        self.user_name = user_name
        self.data = data
        after = [data[i] for i in xrange(0,len(data)) if data[i] >= (datetime.datetime(2014,10,04,)) and data[i] <= (datetime.datetime(2015,10,04))]
        if len(after) == 0: after.append(datetime.datetime(2100,01,01))
        ones = [1]*len(after)
        idx = pd.DatetimeIndex(after)
        serie = pd.Series(ones, index=idx)
        per_week = serie.resample('W', how='count').fillna(0)
        per_week.plot(color="blue")
        #per_week.plot( marker="o", color="green")
        plt.title(user_name+" - A Year After Election")
        plt.ylabel("Number of Posts")
        plt.xlabel("Date (Week)")
        plt.ylim(0,150)
        plt.grid(True, which='both')
        plt.xlim(datetime.datetime(2014,10,05), datetime.datetime(2015,10,04))
        plt.savefig(self.dir_out+self.id_rep+"_"+self.user_name+"_after"".png")
        plt.clf()


if __name__=='__main__':

    dir_in = "/Users/lucasso/Documents/workspace/json/coleta/"
    dir_out = "/Users/lucasso/Documents/workspace/json/coleta/novos/"
    excel_path = "/Users/lucasso/Dropbox/Twitter_Marcelo/Arquivo Principal da Pesquisa - Quatro Etapas.xls"
    sheet_name = "Sheet1"
    col = 6
    pt = PlotTwitter(dir_in, dir_out, excel_path, sheet_name, col )
    pt.run()
