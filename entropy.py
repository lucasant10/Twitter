from text_processor import TextProcessor
from read_twitter import ReadTwitter
from collections import Counter
import itertools
import numpy as np
from apriori import Apriori
import matplotlib.pyplot as plt
import json





def days2time(days):
    #1380844800000  = 04/10/2013, 86400000 = 1 day 
    return 1380844800000+(days*86400000)

def days2timeInterval(day1, day2):
    #1380844800000  = 04/10/2013, 86400000 = 1 day 
    return (1380844800000+(day1*86400000)), (1380844800000+(day2*86400000))

if __name__=='__main__':

    dir_in = "/Users/lucasso/Documents/tweets_pedro/"
    dir_out = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/plot/"
    excel_path = "/Users/lucasso/Dropbox/Twitter_Marcelo/Arquivo Principal da Pesquisa - Quatro Etapas.xls"
    sheet_name = "amostra"
    col = 4
    rt = ReadTwitter(dir_in, excel_path, sheet_name, col )
    tp = TextProcessor()
   

    id_rep, names = rt.names_from_xls()
    parl_words =  Counter()
    counter_list = list()
    tw_apriori = list()

    for idx in range(len(names)):
        tweets = list()
        data = rt.tweets_election_data(id_rep[idx])

        for k,v in data.items():
            tweets.append(tp.text_process(v.split()))

        tw_apriori += [[x[0] for x in e if x]  for e  in tweets if e ]
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

    x = list()
    y = list()
    for k, v in word_ent.items():
        x.append(v)
        y.append(parl_words[k])

    
    fig = plt.figure()
    ax = plt.gca()
    ax.set_yscale('log')
    ax.scatter(x, y,marker="o")
    plt.show()
    """
    filter_list = [k for k, v in word_ent.items() if float(v) == 0]
    list_apriori = [ list(filter(lambda x: x in filter_list, l)) for l in tw_apriori ]

    ap = Apriori(list_apriori, 0.10, 0.2)
    ap.run()
    ap.print_frequent_itemset()
    ap.print_rule()

    """