from text_processor import TextProcessor
import json
from read_twitter import ReadTwitter
from collections import Counter
import itertools
import numpy as np
import os




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
    sheet_name = "reelected"
    col = 4
    rt = ReadTwitter(dir_in, excel_path, sheet_name, col )
    tp = TextProcessor()
   

    dict_list,names = rt.tweets_time_data()

    for idx in range(len(names)):

        weeks = list([0])
        months = list()
        tweets = list()

        data = dict_list[idx]
        dt = {k:v for (k,v) in data.items()}
        #f = list(itertools.chain.from_iterable(b)) 104
        for i in dt:
            dt[i] =  list(itertools.chain.from_iterable(tp.text_process(dt[i].split())))

        for k,v in data.items():
            tweets.append(tp.text_process(v.split()))

        tweets = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(tweets))))
        t_count = Counter(tweets)

        counter = list()
        soma = 0
        n = 0

        """  
        for w, c in t_count.items():
            s = "\n"+w+"|"+str(c)+"|"
            for k in range(0,24):
                inicio,fim = days2timeInterval((k*30), (k+1)*30)
                counter = [Counter(v)[w] for (c,v) in dt.items() if (c >= inicio and c <= fim) ]
                soma = sum(counter)
                if soma >= 1: n+=1
                s+=str(soma)+"|"
            print(s+str(n)+"|")
            n=0

        """ 
        out=""        
        for w,c in t_count.most_common():
            out += "\n"+w+"-|"
            for k in range(0,104):
                inicio,fim = days2timeInterval((k*7), (k+1)*7)
                counter = [Counter(v)[w] for (c,v) in dt.items() if (c >= inicio and c <= fim) ]
                soma = sum(counter)
                if soma >= 1:
                    weeks.append(k+1)
            for i in range(len(weeks)-1):
                out+= str(weeks[i+1]-weeks[i])+"|"
            weeks = [0]
        f =  open(dir_out+names[idx]+".txt", 'w')
        f.write(out)
        f.close()
