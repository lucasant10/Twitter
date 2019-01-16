import sys
sys.path.append('../')
import json
import os
import logging
import pandas as pd
import multiprocessing as mp


try:
    to_unicode = unicode
except NameError:
    to_unicode = str

def create_data_frame(file): 
    user_id = int(file.replace(".json",""))
    dep_tweets = list()
    tmp_df = pd.DataFrame()
    logger.info(">>>>>> processing file: %s" % file)
    try:
        if not df[user_id == df['user_id']].empty:
            logger.info(">>>>>> file: %s is in DF" % file)
            json_data = open(dir_dataset + file).read()
            data = json.loads(json_data)
            tmp_df = pd.DataFrame.from_dict(data,orient='columns')
    except Exception as e:
        print("Unexpected error: {}".format(e))
        logger.exception(e)
    return tmp_df

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('create_df_from_files.log')
    handler.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    # create a logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(consoleHandler)

    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_dataset = path['dir_dataset']

    files = ([file for file in os.listdir(dir_dataset) if file.endswith('.json')])

    workers = (mp.cpu_count()-1)
    logger.info(">>>>>> number of workes: %i" % workers)
    pool = mp.Pool(processes=(workers))
    logger.info(">>>>>> Call create_data_frame with multiprocessing")
    data_frames = pool.map(create_data_frame, files)
    logger.info(">>>>>> Concatenating DF")
    df = pd.concat(data_frames)
    logger.info(">>>>>> Saving DF")
    df.to_pickle(dir_dataset + 'df_dep_tweets.pkl')
    


