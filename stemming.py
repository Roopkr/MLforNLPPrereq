
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer,RegexpStemmer
words=["eating","eats","eaten","writing","writes","programming","programs","history","finally","finalized","Fairly","Sportingly"]

for word in words:
    poerter_stemmer = PorterStemmer()
    lancaster_stemmer = LancasterStemmer()
    snowball_stemmer = SnowballStemmer(language='english')
    regex_stemmer = RegexpStemmer(regexp="ing$|s$|ed$|ly$|ized$",min=4)
    print(f"Original Word: {word}")
    print(f"Porter Stemmer: {poerter_stemmer.stem(word)}")
    print(f"Lancaster Stemmer: {lancaster_stemmer.stem(word)}")
    print(f"Snowball Stemmer: {snowball_stemmer.stem(word)}")
    print(f"Regex Stemmer: {regex_stemmer.stem(word)}")