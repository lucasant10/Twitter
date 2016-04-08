import os
import json
import xlrd



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
        for i in range(sheet.nrows):
            id_rep = str(int(sheet.cell_value(rowx= i, colx=self.col)))
            if (id_rep in self.rep_dic):
                with open(self.dir_in+self.rep_dic[id_rep]) as data_file:
                    for line in data_file:
                        tweet = json.loads(line)
                        doc_set.add(tweet['text'])
        return doc_set

    def create_dic(self, fnames):
        for fname in fnames:
            self.rep_dic[fname.split('_',1)[0]] = fname
                       