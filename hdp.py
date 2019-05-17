from gensim import corpora
from gensim.models import HdpModel
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re

stop_words = set(stopwords.words('english')) 

file = open("ap_changed.txt","r+")
documents = file.readlines()

#word_tokens = word_tokenize([for doc in documents]) 

texts =  [[re.sub(r'[^a-z]', '', text.lower()) for text in doc.split() if not text in stop_words and re.sub(r'[^a-z]', '', text)] for doc in documents]
#print('TEXT: ', texts[0])

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)

print(hdpmodel.print_topics(1000, 1))

#print(hdptopics)