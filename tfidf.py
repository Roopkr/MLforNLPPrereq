import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('Notebooks/smsspamcollection/SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

corpus = []
for sentence in df['message']:
    sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    sentence = sentence.lower()
    final_words = [word for word in sentence.split() if word not in stopwords.words('english')]
    sentence =  ' '.join(final_words)
    corpus.append(sentence)

tfid = TfidfVectorizer(max_features=100, ngram_range=(1,1))
X_tfidf = tfid.fit_transform(corpus).toarray()
print("Shape of the TF-IDF model array:", X_tfidf.shape)
print("X_tfidf", X_tfidf)
print("Vocabulary:")
print(tfid.vocabulary_)
