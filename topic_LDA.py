import sys
import configparser
from gensim import corpora, matutils
import gensim
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Create LDA Model')
    parser.add_argument('-f', '--filename', required=True, help='Text file')
    parser.add_argument('-o', '--output', required=True, help='LDA model file name')
    parser.add_argument('-n', '--num_topics', required = True)

    args = parser.parse_args()
    filename = args.filename
    output = args.output
    num_topics = int(args.num_topics)

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_lda = path['dir_lda']

    print("Loading file %s " % filename )
    texts = list()
    with open(dir_lda + "doc/" + filename, 'r') as data_file:
        for line in data_file:
            texts.append(line.split())

    print("Creating dictionary")
    dictionary = corpora.Dictionary(texts)
    dictionary.compactify()
    # and save the dictionary for future use
    #dictionary.save('tweet_teste.dict') 

    # convert tokenized documents into a document-term matrix
    print("Creating corpus")
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    print("Processing LDA")
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary,alpha='auto')

    print(ldamodel.print_topics())
    ldamodel.save(dir_lda+output)
    #model = gensim.models.LdaModel.load('android.lda')