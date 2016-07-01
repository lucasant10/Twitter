from text_processor import TextProcessor
from read_twitter import ReadTwitter
from collections import Counter
import itertools
import numpy as np
import json





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
    parl_words =  Counter()
    counter_list = list()

    for idx in range(len(names)):
        tweets = list()
        data = rt.tweets_election_data(id_rep[idx])

        for k,v in data.items():
            tweets.append(tp.text_process(v.split()))

        tweets = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(tweets))))
        counter_list.append(Counter(tweets))
        parl_words.update(tweets)

    word_ent = dict()
    for word,count in parl_words.items():
        ent = 0
        for counter in counter_list:
            prob = counter[word]/count
            ent += prob * (-np.log2(prob+1e-100))
        word_ent[word] = ent
    sort = sorted(word_ent.items(), key=lambda x: x[1], reverse=True)
    with open(dir_out+"entropy.txt", "w") as f: f.write(json.dumps(sort))
