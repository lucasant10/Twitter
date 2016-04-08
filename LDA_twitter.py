from pymongo import MongoClient
from nltk.tokenize import RegexpTokenizer


# connect to the MongoDB
client      = MongoClient()
db          = client['parlamentares']


documents = [re.sub(r"(?:\@|http?\://)\S+", "", doc)
                for doc in documents ]

tokenizer = RegexpTokenizer(r'\w+')
documents = [ tokenizer.tokenize(doc.lower()) for doc in documents ]

stoplist  = set(nltk.corpus.stopwords.words("portuguese") )
documents = [[token for token in doc if token not in stoplist]
                for doc in documents]

# rm numbers only words
documents = [ [token for token in doc if len(token.strip(digits)) == len(token)]
                for doc in documents ]

# Remove words that only occur once
token_frequency = defaultdict(int)

# count all token
for doc in documents:
    for token in doc:
        token_frequency[token] += 1


# Sort words in documents
for doc in documents:
    doc.sort()

dictionary = corpora.Dictionary(documents)
dictionary.compactify()
# and save the dictionary for future use
dictionary.save('tweet_teste.dict')

# We now have a dictionary with 26652 unique tokens
print(dictionary)


# Build the corpus: vectors with occurence of each word for each document
# convert tokenized documents to vectors
corpus = [dictionary.doc2bow(doc) for doc in documents]

# and save in Market Matrix format
corpora.MmCorpus.serialize('tweet_teste.mm', corpus)
# this corpus can be loaded with corpus = corpora.MmCorpus('alexip_followers.mm')

