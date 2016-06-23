import os
import xlrd
import json
import csv 

class Mapa:

    rep_dic = dict()

    def __init__(self, dir_out, dir_in, excel_path):
        self.dir_out = dir_out
        self.dir_in = dir_in
        self.excel_path = excel_path
        fnames = ([file for root, dirs, files in os.walk(self.dir_in)
            for file in files if file.endswith('.json')  ])
        self.create_dic(fnames)

    def create_dic(self, fnames):
        for fname in fnames:
            self.rep_dic[fname.split('_',1)[0]] = fname

    def save_profile(self, sheet_name):
        xls = xlrd.open_workbook(self.excel_path)
        sheet = xls.sheet_by_name(sheet_name)
        dic = dict()
        dic_rt = dict()
        for i in range(len(sheet.col(0))):
            id_rep = str(int(sheet.cell_value(rowx= i, colx=4)))
            #nome = sheet.cell_value(rowx= i, colx=6)
            estado = sheet.cell_value(rowx= i, colx=7)
            if(not estado in dic):
                dic[estado] = 0
                dic_rt[estado] = 0
            if (id_rep in self.rep_dic):
                with open(self.dir_in+self.rep_dic[id_rep]) as data_file:  
                    num = 0
                    rt = 0
                    for line in data_file:
                        tweet = json.loads(line)
                        num += 1
                        rt += int(tweet['retweets'])
                    dic[estado] += num
                    dic_rt[estado] += rt
        with open(self.dir_out+'retweet.txt', 'w') as f:
            f.writelines('{}:{}\n'.format(k,v) for k, v in dic_rt.items())
            f.close()
        with open(self.dir_out+'tweet.txt', 'w') as fa:
            fa.writelines('{}:{}\n'.format(k,v) for k, v in dic.items())
            fa.close()



if __name__=='__main__':

    dir_out = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/profiles/"
    dir_in = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/"
    excel_path = "/Users/lucasso/Dropbox/Twitter_Marcelo/Arquivo Principal da Pesquisa - Quatro Etapas.xls"
    sheet_name = "novos"
    pt = Mapa(dir_out, dir_in, excel_path)
    pt.save_profile(sheet_name)