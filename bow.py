import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_csv('Notebooks/smsspamcollection/SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
print("First 5 rows of the dataset:")
print(df.head())
corpus = []
corpus_lemmatized = []
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
for sentence in df['message']:
    sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    sentence = sentence.lower()
    final_words = [ps.stem(word) for word in sentence.split() if word not in stopwords.words('english')]
    sentence =  ' '.join(final_words)
    corpus.append(sentence)

for sentence in df['message']:
    sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    sentence = sentence.lower()
    final_words = [lemmatizer.lemmatize(word,pos='v') for word in sentence.split() if word not in stopwords.words('english')]
    sentence =  ' '.join(final_words)
    corpus_lemmatized.append(sentence)

print("First 5 preprocessed messages in the corpus:")
for i in range(5): 
    print(f"Message {i+1}: {corpus[i]}")

cv = CountVectorizer(max_features=100,ngram_range=(1,2))
X = cv.fit_transform(corpus).toarray()
lX = cv.fit_transform(corpus_lemmatized).toarray()
print("Shape of the Bag of Words model array:", X.shape)
print("Shape of the Lemmatized Bag of Words model array:", lX.shape)
print("X",X)
print("lX",lX)
print("Feature names in the Bag of Words model:")
print(cv.get_feature_names_out())
print("Features:")
print(cv.vocabulary_)
