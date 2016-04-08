import os
import json
from matplotlib import pyplot as plt
import datetime
import pandas as pd
from matplotlib import dates
import xlrd
from matplotlib import numpy as np
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('utf-8')
import dabase


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
        self.save_db()
        print("No file for these Ids:")
        print(self.erro)
        np.savetxt(self.dir_out+'Ids_not_found.txt', self.erro, delimiter=",", fmt="%s")

    def save(self):

        xls = xlrd.open_workbook(self.excel_path)
        sheet = xls.sheet_by_name(self.sheet_name)

        for i in xrange(sheet.nrows):
            id_rep = str(int(sheet.cell_value(rowx= i, colx=self.col)))
            user_name=""
            date_fq =[]
            if (self.rep_dic.has_key(id_rep)):
                with open(self.dir_in+self.rep_dic[id_rep]) as data_file:
                    for line in data_file:
                        tweet = json.loads(line)
                        date_fq.append(pd.to_datetime(tweet['created_at']*1000000))
                        user_name = tweet['user_name']
                print("processing: "+user_name+" id "+id_rep)
               # self.save_before(id_rep, user_name,date_fq)
                #self.save_after(id_rep, user_name,date_fq)
                self.cumulative_hist(id_rep, user_name, date_fq)
                #self.save_all(id_rep, user_name,date_fq)
            else:
                self.erro.append(id_rep)

    def save_db(self):

            xls = xlrd.open_workbook(self.excel_path)
            sheet = xls.sheet_by_name(self.sheet_name)
            for i in xrange(sheet.nrows):
                id_rep = str(int(sheet.cell_value(rowx= i, colx=self.col)))
                condicao = str(sheet.cell_value(rowx= i, colx=self.col+1))
                user_name=""
                if (self.rep_dic.has_key(id_rep)):
                    with open(self.dir_in+self.rep_dic[id_rep]) as data_file:
                        for line in data_file:
                            tweet = json.loads(line)
                            date = pd.to_datetime(tweet['created_at']*1000000)
                            user_name = tweet['user_name']
                            tw = dabase.Twitter(user_name=user_name, id_parlamentar=id_rep, condicao=condicao, tweet_data=date.strftime("%y-%m-%d"), tweet_hora=date.strftime("%H:%M:%S"))
                            tw.save()


                    print("processing: "+user_name+" id "+id_rep)
                    


                else:
                    self.erro.append(id_rep)

    def create_dic(self, fnames):
        for fname in fnames:
            self.rep_dic[fname.split('_',1)[0]] = fname



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
        per_week.plot(figsize=(9,7),color="blue")
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
        per_week.plot(figsize=(9,7),color="red")
        #per_week.plot( marker="o", color="green")
        plt.title(user_name+" - A Year After Election")
        plt.ylabel("Number of Posts")
        plt.xlabel("Date (Week)")
        plt.ylim(0,150)
        plt.grid(True, which='both')
        #plt.axvspan('10/04/2014','10/04/2014', color='blue')
        plt.xlim(datetime.datetime(2014,10,04), datetime.datetime(2015,10,04))
        #plt.savefig(self.dir_out+self.id_rep+"_"+self.user_name+"_after"".png")
        plt.clf()


    def save_all(self, id_rep, user_name, data):
        self.id_rep = id_rep
        self.user_name = user_name
        self.data = data
        whole_period = [data[i] for i in xrange(0,len(data)) if data[i] >= (datetime.datetime(2014,10,04,)) and data[i] <= (datetime.datetime(2015,10,04))]
        if len(whole_period) == 0: whole_period.append(datetime.datetime(2100,01,01))
        ones = [1]*len(whole_period)
        idx = pd.DatetimeIndex(whole_period)
        serie = pd.Series(ones, index=idx)
        per_week = serie.resample('W', how='count').fillna(0)
        per_week.plot(figsize=(9,7),color="red")
        #per_week.plot( marker="o", color="green")
        plt.title(user_name+" - A Year Before and After Election")
        plt.ylabel("Number of Posts")
        plt.xlabel("Date (Week)")
        plt.ylim(0,150)
        plt.grid(True, which='both')
        plt.axvspan('10/04/2014','10/04/2014', color='blue')

        plt.xlim(datetime.datetime(2013,10,04), datetime.datetime(2015,10,04))
        plt.show()
        #plt.savefig(self.dir_out+self.id_rep+"_"+self.user_name+"_after"".png")
        plt.clf()

    def cumulative_hist(self, id_rep, user_name, data):
        self.id_rep = id_rep
        self.user_name = user_name
        self.data = data
        after = [data[i] for i in xrange(0,len(data)) if data[i] >= (datetime.datetime(2013,10,04,)) and data[i] <= (datetime.datetime(2014,10,04))]
        if len(after) == 0: after.append(datetime.datetime(2100,01,01))
        plt.title(user_name+" - A Year Before Election ")
        plt.hist(dates.date2num(after), cumulative=True)
        plt.savefig(self.dir_out+self.id_rep+"_"+self.user_name+"_before"".png")
        plt.show()
        plt.clf()


if __name__=='__main__':

    dir_in = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/"
    dir_out = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/plot/"
    excel_path = "/Users/lucasso/Dropbox/Twitter_Marcelo/Arquivo Principal da Pesquisa - Quatro Etapas.xls"
    sheet_name = "nao_eleitos"
    col = 4
    pt = PlotTwitter(dir_in, dir_out, excel_path, sheet_name, col )
    pt.run()
  