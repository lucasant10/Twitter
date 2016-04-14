from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
#from stop_words import get_stop_words
from gensim import corpora
import gensim
import re
from read_twitter import ReadTwitter
from unicodedata import normalize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json



class TextProcess:

    tokenizer = RegexpTokenizer(r'\w+')

    stoplist  = stopwords.words("portuguese")
    #stoplist = get_stop_words('portuguese')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = SnowballStemmer("portuguese")

    # remvove urls

    def remove_urls(self, txt):  
        txt = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', txt)
        return txt


    def remover_acentos(self, txt):
        return normalize('NFKD', txt).encode('ASCII','ignore').decode('ASCII')

    def plot_text(self,texts, name):
        txt = ""
        tokens = self.text_process(texts,False)
        for text in tokens :
            for word in text:
                txt += " "+word
        wc = WordCloud().generate(txt)
        plt.imshow(wc)
        plt.savefig('./img/'+name+'.png', dpi=300)

    def text_process(self,doc_set, stem=True):
        # list for tokenized documents in loop
        texts = []
        # loop through document list
        for i in doc_set:
            # clean and tokenize document string
            raw = i.lower()

            #remove urls
            raw = self.remove_urls(raw) 
            #raw = remover_acentos(raw)
    
            tokens = self.tokenizer.tokenize(raw)
          
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in self.stoplist]

            #remove unigrams and bigrams
            stopped_tokens = [i for i in stopped_tokens if len(i) > 2]
            # remove acentos
            stopped_tokens = [self.remover_acentos(i) for i in stopped_tokens]
            
            if stem:
                # stem tokens
                stopped_tokens = [self.p_stemmer.stem(i) for i in stopped_tokens]
            
            # add tokens to list
            texts.append(stopped_tokens)
        return texts

    def create_corpus(texts):
        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(texts)
        dictionary.compactify()
        # and save the dictionary for future use
        #dictionary.save('tweet_teste.dict')

            
        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in texts]

        # and save in Market Matrix format
        #corpora.MmCorpus.serialize('tweet_teste.mm', corpus)
        # this corpus can be loaded with corpus = corpora.MmCorpus('tweet_teste.mm')
        return corpus


    def  generate_lda(corpus, dictionary, num_topics):
            
        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary)
        ldamodel.save('tweet_teste.lda')
        #model = gensim.models.LdaModel.load('android.lda')
        print(ldamodel.print_topics())
        ldamodel.print_topics()


if __name__=='__main__':

    dir_in = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/"
    excel_path = "/Users/lucasso/Dropbox/Twitter_Marcelo/Arquivo Principal da Pesquisa - Quatro Etapas.xls"
    sheet_name = "novos"
    col = 4
    rt = ReadTwitter(dir_in, excel_path, sheet_name, col )
    tp = TextProcess()
    name, doc =  rt.tweets_by_rep()

    for i in range(len(name)):
        tp.plot_text(doc[i],name[i])
        #doc_set = set()
        #doc_set = rt.tweets()
        """
        filedir = "/Users/lucasso/Dropbox/Twitter_Marcelo/Report/coleta_pedro/74171_Chico Alencar.json"
        with open(filedir) as data_file:
            for line in data_file:
                tweet = json.loads(line)
                doc_set.add(tweet['text'])
        """

