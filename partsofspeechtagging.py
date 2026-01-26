import nltk
from nltk.tokenize import sent_tokenize

nltk.download('averaged_perceptron_tagger_eng')
corpus = """Natural language processing (NLP) is a fascinating field of study that focuses on the interaction between computers and human (natural) languages. It involves the development of algorithms and models that enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful. NLP has a wide range of applications, including machine translation, sentiment analysis, chatbots, and information retrieval. By leveraging techniques from linguistics, computer science, and artificial intelligence, NLP aims to bridge the gap between human communication and computer understanding, making it possible for machines to process and analyze vast amounts of textual data efficiently."""
sentence_tokens = sent_tokenize(corpus)
for word in sentence_tokens:
    pos_tag = nltk.pos_tag(word.split())
    print(f"Sentence: {word}")
    print(f"Parts of Speech Tagging: {pos_tag}")

sentence = "The quick brown fox jumps over the lazy dog."
print(f"Sentence: {sentence}")
print(nltk.pos_tag(sentence.split()))

