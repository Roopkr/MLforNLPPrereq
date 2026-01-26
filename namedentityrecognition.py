import nltk
from nltk.tokenize  import word_tokenize

nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
sentence = "The Eiffel Tower was built from 1887 to 1889 by Gustave Eiffel, whose company specialized in building metal frameworks and structures."
postag = nltk.pos_tag(sentence.split())
nltk.ne_chunk(postag).draw()
print(nltk.ne_chunk(postag))
