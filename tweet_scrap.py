import twitterscraper
import datetime as dt
import json
import logging
import pandas as pd
import collections
import requests
import query
from time import sleep

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__json__'):
            return obj.__json__()
        elif isinstance(obj, collections.Iterable):
            return list(obj)
        elif isinstance(obj, dt.datetime):
            return obj.isoformat()
        elif hasattr(obj, '__getitem__') and hasattr(obj, 'keys'):
            return dict(obj)
        elif hasattr(obj, '__dict__'):
            return {member: getattr(obj, member)
                    for member in dir(obj)
                    if not member.startswith('_') and
                    not hasattr(getattr(obj, member), '__call__')}

        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    logging.basicConfig(filename='contas_scrap.log', level=logging.INFO)
    logging.info('Started')
    df= pd.read_csv('contas/deputados_2.csv')
    for conta in df.conta.values:
        try:
            logging.info("==== "+conta+" ====")
            tweets = query.query_tweets('from:%s'%conta, begindate=dt.date(2018, 12, 1), enddate=dt.date(2019, 12, 31))
            if tweets:
                with open("%s.json" % conta, "w", encoding="utf-8") as output:
                    json.dump(tweets, output, cls=JSONEncoder)    
                output.close()
            else:
                logging.info('No Tweets')
        except ValueError as e:
            print(e.message)
            logger.exception("got an error: %s" % e.message)
            output.close()
        sleep(4)
    logging.info('Finished')
     
    


