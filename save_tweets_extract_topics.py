import os
from text_processor import TextProcessor
import json
import itertools
import pickle
import configparser
import re


def load_files(dir_in):
    time_list=list()
    tw_files = ([file for root, dirs, files in os.walk(dir_in)
        for file in files if file.endswith('.json') ])
    parl_tw_list = list()
    for tw_file in tw_files:
        temp=list()
        with open(dir_in+tw_file) as data_file:
            for line in data_file:
                tweet = json.loads(line)
                temp.append(tweet['text'])
                time_list.append(tweet['created_at'])
        parl_tw_list.append(temp)
    return time_list, parl_tw_list 

def process_dict(time_list,parl_tw_list):
    tp = TextProcessor()
    processed_list = list()
    time_doc_list = list()
    for parl_tweets in parl_tw_list:
        tmp = tp.text_process(parl_tweets, text_only=True)
        tmp =  [" ".join(x) for x in tmp]
        processed_list.append(tmp)
    tweets = list(itertools.chain.from_iterable(parl_tw_list))
    print("saiu do processamento de texto")
    for i in range(len(time_list)):
        time_doc_list.append((time_list[i],tweets[i]))
    return time_doc_list,processed_list

def replace_ngram(ngram,sentence):
    word = str.replace(ngram,"_", " ")
    s = re.sub(r'\b%s\b' % word,ngram,sentence)
    return s

def save_file(path,file_name,bgr_list):
    with open(dir_out+path+file_name+".txt", "w") as f:
        [f.write(" ".join(x)+"\n") for x in bgr_list if x]
        f.close()

def days2time(days):
    #1235952000000  = 03/10/2013, 86400000 = 1 day 
    return 1235952000000+(days*86400000)


if __name__=='__main__':

cf = configparser.ConfigParser()
cf.read("file_path.properties")
path = dict(cf.items("file_path"))
dir_in = path['dir_in']
dir_out = path['dir_out']
dir_pck = path['dir_pck']

time_list, parl_tw_list = load_files(dir_in)
time_doc_list, parl_tw_list = process_dict(time_list,parl_tw_list)

with open(dir_out+"joint_parl_tw_list.pck", 'wb') as handle:
    pickle.dump(parl_tw_list, handle)

pck = ([file for root, dirs, files in os.walk(dir_pck)
        for file in files if file.endswith('.pck') ])

ranked_list = list()
for p in pck:
    with open(dir_pck+p,'rb') as handle:
        rank = pickle.load(handle)
    ranked_list.append([" ".join(i) for i,v in rank[:10000]])
    ranked_list.append(["_".join(i) for i,v in rank[:5000]])
    ranked_list.append(["_".join(i) for i,v in rank[:2000]])
    ranked_list.append(["_".join(i) for i,v in rank[:1000]])

doc_like_list = list()
doc_beta_list = list()
os.makedirs(dir_out+"tw_bgr/10k_beta")
os.makedirs(dir_out+"tw_bgr/5k_beta")
os.makedirs(dir_out+"tw_bgr/2k_beta")
os.makedirs(dir_out+"tw_bgr/1k_beta")
os.makedirs(dir_out+"tw_bgr/10k_like")
os.makedirs(dir_out+"tw_bgr/5k_like")
os.makedirs(dir_out+"tw_bgr/2k_like")
os.makedirs(dir_out+"tw_bgr/1k_like")

for i, parl_tw in enumerate(parl_tw_list):
    beta_tmp = list()
    like_tmp = list()
    print("processing parl: "+str(i))
    for tweet in parl_tw:
        beta_tmp.append([ str.replace(x," ","_") for x in ranked_list[0] if x in tweet])
        like_tmp.append([ str.replace(x," ","_") for x in ranked_list[4] if x in tweet])
    doc_beta_list.append(beta_tmp)
    doc_like_list.append(like_tmp)
    print("saving files beta")    
    save_file("tw_bgr/10k_beta/",str(i),beta_tmp)
    save_file("tw_bgr/5k_beta/",str(i),[[x for x in ranked_list[1] if x in y] for y in beta_tmp])
    save_file("tw_bgr/2k_beta/",str(i),[[x for x in ranked_list[2] if x in y] for y in beta_tmp])
    save_file("tw_bgr/1k_beta/",str(i),[[x for x in ranked_list[3] if x in y] for y in beta_tmp])
    print("saving files like")
    save_file("tw_bgr/10k_like/",str(i),like_tmp)
    save_file("tw_bgr/5k_like/",str(i),[[x for x in ranked_list[5] if x in y] for y in like_tmp])
    save_file("tw_bgr/2k_like/",str(i),[[x for x in ranked_list[6] if x in y] for y in like_tmp])
    save_file("tw_bgr/1k_like/",str(i),[[x for x in ranked_list[7] if x in y] for y in like_tmp])
        
time_doc_beta_list = list()
time_doc_like_list = list()

tweets_beta = list(itertools.chain.from_iterable(doc_beta_list))
tweets_like = list(itertools.chain.from_iterable(doc_like_list))
print("saiu do processamento de texto")
for i in range(len(time_list)):
    time_doc_beta_list.append((time_list[i],tweets_beta[i]))
    time_doc_like_list.append((time_list[i],tweets_like[i]))

with open(dir_out+"time_doc_beta_list.pck", 'wb') as handle:
    pickle.dump(time_doc_beta_list, handle)

with open(dir_out+"time_doc_like_list.pck", 'wb') as handle:
    pickle.dump(time_doc_like_list, handle)
    
time_doc_beta_list = [x for x in time_doc_beta_list if x[1]]
time_doc_like_list = [x for x in time_doc_like_list if x[1]]

first = 1224550880000
last = 1454803200000
week = 604800000 

for w in range(first, last-week, week):
    w_number = int(((w-first)/week)+1)
    temp = [x for x in time_doc_beta_list if (int(x[0]) > w and int(x[0]) <= (w+week))]
    save_file("tw_week_bgr/","week_"+str(w_number)+"_beta_10k",[x[1] for x in temp])
    save_file("tw_week_bgr/","week_"+str(w_number)+"_beta_5k",[[x for x in ranked_list[1] if x in y[1]] for y in temp])
    save_file("tw_week_bgr/","week_"+str(w_number)+"_beta_2k",[[x for x in ranked_list[2] if x in y[1]] for y in temp])
    save_file("tw_week_bgr/","week_"+str(w_number)+"_beta_1k",[[x for x in ranked_list[3] if x in y[1]] for y in temp])
    temp = [x for x in time_doc_like_list if (int(x[0]) > w and int(x[0]) <= (w+week))]
    save_file("tw_week_bgr/","week_"+str(w_number)+"_like_10k",[x[1] for x in temp])
    save_file("tw_week_bgr/","week_"+str(w_number)+"_like_5k",[[x for x in ranked_list[5] if x in y[1]] for y in temp])
    save_file("tw_week_bgr/","week_"+str(w_number)+"_like_2k",[[x for x in ranked_list[6] if x in y[1]] for y in temp])
    save_file("tw_week_bgr/","week_"+str(w_number)+"_like_1k",[[x for x in ranked_list[7] if x in y[1]] for y in temp])
    w += week
    
    


def keyFunc(afilename):
    nondigits = re.compile("\D")
    return int(nondigits.sub("", afilename))
    
pck = ([file for root, dirs, files in os.walk(dir_out+"tw_week_bgr/")
    for file in files if file.endswith('beta_10k.txt') ])
pck  = sorted(pck, key=keyFunc)

inicial =0
final = 361 
tmp = list()
for k in range(inicial, final, 4):
    tmp.append(pck[k:k+4])

for i, m in enumerate(tmp):
    txt = ""
    for wk in m:
        for line in open(dir_out+"tw_week_bgr/"+wk,"rb"):
            txt += line.decode('utf-8')
    f =  codecs.open(dir_out+"tw_week_bgr/month/"+"month"+str(i)+"_beta_10k.txt", 'w',encoding='utf-8')
    f.write(txt)
    f.close()



pck = ([file for root, dirs, files in os.walk(dir_w2v)
    for file in files if file.endswith('.txt') ])

doc_list = list()

for m in pck:
    dp = []
    with open(dir_w2v+m,"rb") as data_file:
        for line in data_file:
            dp.append(line.decode('utf-8'))
    doc_list.append(dp)    

doc = [dir_out+"antes/"+x for x in pck]

#1235952000000
#1454803200000

with open(dir_out+"time_doc_parl_processed.pck",'rb') as handle:
    parl_list = pickle.load(handle)

for 

#mes antes
#1412467200 eleicao
#1415145600 mes depois






