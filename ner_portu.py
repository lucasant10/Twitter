import nltk
from os.path import join


def simplify_tag(t):
    if "+" in t: t = t[t.index("+")+1:]
    if "|" in t: t = t[t.index("|")+1:]
    if "#" in t: t = t[0:t.index("#")]
    t = t.lower()
    return tagmap[t]

dataset1 = nltk.corpus.floresta.tagged_sents( )

dataset2 = nltk.corpus.mac_morpho.tagged_sents( ) 

traindata = dataset1 + dataset2 
traindata = [ [ ( w , simplify_tag(t) ) for ( w , t ) in sent ] for sent in traindata ]

tagger_fast = nltk.NgramTagger(4, traindata, backoff=nltk.TrigramTagger(traindata, backoff=nltk.BigramTagger(traindata, backoff=nltk.UnigramTagger(traindata, backoff=nltk.DefaultTagger('NOUN')))))

pos_result = tagger_fast.tag( nltk.tokenize.word_tokenize("E agora para uma coisa completamente diferente! Viva a Maria!") )
print pos_result

pattern = """ NP: {<PRON|ADP|DET>?<ADJ>*<NOUN>+} """
chunker_fast = nltk.RegexpParser(pattern)

chunk_result = chunker_fast.parse(pos_result)
print chunk_result