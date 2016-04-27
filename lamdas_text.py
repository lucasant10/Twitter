from text_processor import TextProcessor
import json

def days2time(days):
    #1380844800000  = 04/10/2013, 86400000 = 1 day 
    return 1380844800000+(days*86400000)
if __name__=='__main__':

    dir_out = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/plot/img/"
    filedir = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/178916_Daniel Coelho.json"
    tp = TextProcessor()

    with open(filedir) as data_file:
        doc_set = set()
        for line in data_file:
            tweet = json.loads(line)
            created = int(tweet['created_at'])
            if(days2time(561) <= created < days2time(575)):
                doc_set.add(tweet['text'])
        tp.plot_text(doc_set, str(561)+"Daniel Coelho", dir_out)
