from text_processor import TextProcessor
import json

def days2time(days):
    #1380844800000  = 04/10/2013, 86400000 = 1 day 
    return 1380844800000+(days*86400000)
if __name__=='__main__':

    dir_out = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/plot/weeks/"
    filedir = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/160640_Eros Biondini.json"
    tp = TextProcessor()

    with open(filedir) as data_file:
        doc_set = list()
        inicial = 155
        final = 197
        for line in data_file:
            tweet = json.loads(line)
            created = int(tweet['created_at'])
            if(days2time(inicial) <= created < days2time(final)):
                doc_set.append(tweet)
        for k in range(inicial, final, 7):
            doc = set()
            for tw in doc_set:
                if(days2time(k) <= tw['created_at'] < days2time(k+7)):
                    doc.add(tw['text'])
            tp.plot_text(doc, str(k)+"Eros Biondini", dir_out)
