from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
#from stop_words import get_stop_words
from gensim import corpora
import gensim
import re
from read_twitter import ReadTwitter
from unicodedata import normalize

dir_in = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/"
excel_path = "/Users/lucasso/Dropbox/Twitter_Marcelo/Arquivo Principal da Pesquisa - Quatro Etapas.xls"
sheet_name = "reeleitos"
col = 4
rt = ReadTwitter(dir_in, excel_path, sheet_name, col )
#doc_set = set()
doc_set = rt.tweets()

"""
filedir = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/62881_Danilo Forte.json"
with open(filedir) as data_file:
    for line in data_file:
        tweet = json.loads(line)
        doc_set.add(tweet['text'])
"""
tokenizer = RegexpTokenizer(r'\w+')

stoplist  = set(stopwords.words("portuguese") )
#stoplist = get_stop_words('portuguese')

# Create p_stemmer of class PorterStemmer
p_stemmer = SnowballStemmer("portuguese")

# remvove urls

def remove_urls(text):
    text = re.sub(r"(?:\@|http?\://)\S+", "", text)
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    return text


def remover_acentos(txt):
    return normalize('NFKD', txt).encode('ASCII','ignore').decode('ASCII')

# list for tokenized documents in loop
texts = []
# loop through document list
for i in doc_set:
    # clean and tokenize document string
    raw = i.lower()
    #remove urls
    raw = remove_urls(raw) 
    #raw = remover_acentos(raw)
    raw = p_stemmer.stem(raw)
    
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in stoplist]
    #remove unigrams and bigrams
    stopped_tokens = [i for i in tokens if len(i) > 2]
    # remove acentos
    stopped_tokens = [remover_acentos(i) for i in stopped_tokens]
    
    # stem tokens
    #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stopped_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
dictionary.compactify()
# and save the dictionary for future use
dictionary.save('tweet_teste.dict')

    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# and save in Market Matrix format
corpora.MmCorpus.serialize('tweet_teste.mm', corpus)
# this corpus can be loaded with corpus = corpora.MmCorpus('tweet_teste.mm')


# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary)
ldamodel.save('tweet_teste.lda')
print(ldamodel.print_topics())
ldamodel.print_topics()

