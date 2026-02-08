import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.downloader import load

w2v = load('word2vec-google-news-300')

print(w2v['king'])
print(w2v.most_similar('blackman', topn=5))
vec = w2v['king'] - w2v['man'] + w2v['woman']
print(w2v.most_similar([vec], topn=5))
print(w2v.similarity('king', 'queen'))