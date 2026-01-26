import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
words = ["running", "ran", "easily", "fairly", "better", "best", "cats", "geese", "mice", "children"]
lemmatizer = WordNetLemmatizer()
for word in words:
    lemma = lemmatizer.lemmatize(word)
    lemma_verb = lemmatizer.lemmatize(word, pos='v')
    print(f"Original Word: {word}")
    print(f"Lemma (default noun): {lemma}")
    print(f"Lemma (verb): {lemma_verb}")