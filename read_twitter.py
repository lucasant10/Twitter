import os
import json
import xlrd
import pandas as pd
import datetime



class ReadTwitter:

    rep_dic={}

    def __init__(self, dir_in,excel_path, sheet_name, col):
        self.dir_in = dir_in
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.col = col
        fnames = ([file for root, dirs, files in os.walk(self.dir_in)
            for file in files if file.endswith('.json')  ])
        self.create_dic(fnames)
        
    def tweets(self):

        xls = xlrd.open_workbook(self.excel_path)
        sheet = xls.sheet_by_name(self.sheet_name)
        doc_set = set()
        for i in range(len(sheet.col(0))):
            id_rep = str(int(sheet.cell_value(rowx= i, colx=self.col)))
            if (id_rep in self.rep_dic):
                with open(self.dir_in+self.rep_dic[id_rep]) as data_file:
                    for line in data_file:
                        tweet = json.loads(line)
                        doc_set.add(tweet['text'])
        return doc_set

    def tweets_by_rep(self):
       
        xls = xlrd.open_workbook(self.excel_path)
        sheet = xls.sheet_by_name(self.sheet_name)
        name = list()
        doc = list()
        for i in range(len(sheet.col(0))):
            id_rep = str(int(sheet.cell_value(rowx= i, colx=self.col)))
            if (id_rep in self.rep_dic):               
                with open(self.dir_in+self.rep_dic[id_rep]) as data_file:
                    doc_list= list()
                    for line in data_file:
                        tweet = json.loads(line)
                        doc_list.append(tweet['text'])
                    doc.append(doc_list)
                    name.append(self.rep_dic[id_rep].split('.',1)[0])
        return (name, doc)

    def tweets_before_after(self):

        xls = xlrd.open_workbook(self.excel_path)
        sheet = xls.sheet_by_name(self.sheet_name)
        before = set()
        after = set()
        for i in range(sheet.nrows):
            id_rep = str(int(sheet.cell_value(rowx= i, colx=self.col)))
            if (id_rep in self.rep_dic):
                with open(self.dir_in+self.rep_dic[id_rep]) as data_file:
                    for line in data_file:
                        tweet = json.loads(line)
                        date = pd.to_datetime(tweet['created_at']*1000000)
                        if(date <= datetime.datetime(2014,10,4)):
                            before.add(tweet['text'])
                        if(date > datetime.datetime(2014,10,4)):
                            after.add(tweet['text'])
        return (before, after)

    def tweets_after(self):

        xls = xlrd.open_workbook(self.excel_path)
        sheet = xls.sheet_by_name(self.sheet_name)
        doc_set = set()
        for i in len(sheet.col(0)):
            id_rep = str(int(sheet.cell_value(rowx= i, colx=self.col)))
            if (id_rep in self.rep_dic):
                with open(self.dir_in+self.rep_dic[id_rep]) as data_file:
                    for line in data_file:
                        tweet = json.loads(line)
                        date = pd.to_datetime(tweet['created_at']*1000000)
                        if(date > datetime.datetime(2014,10,4)):
                            doc_set.add(tweet['text'])
        return doc_set

    def tweets_election_data(self, id_rep):

        data = dict()

        if (id_rep in self.rep_dic):
            with open(self.dir_in+self.rep_dic[id_rep]) as data_file:
                for line in data_file:
                    tweet = json.loads(line)
                    date = pd.to_datetime(tweet['created_at']*1000000)
                    if(datetime.datetime(2013,10,4) <= date <= datetime.datetime(2015,10,4)):
                        data[int(tweet['created_at'])] = tweet['text']
        return data

    def names_from_xls(self):
        xls = xlrd.open_workbook(self.excel_path)
        sheet = xls.sheet_by_name(self.sheet_name)
        names = list()
        id_rep = list()
        for i in range(sheet.nrows):
            names.append(str(sheet.cell_value(rowx= i, colx=0)))
            id_rep.append(str(int(sheet.cell_value(rowx= i, colx=self.col))))
        return (id_rep,names)

    def create_dic(self, fnames):
        for fname in fnames:
            self.rep_dic[fname.split('_',1)[0]] = fname
                       