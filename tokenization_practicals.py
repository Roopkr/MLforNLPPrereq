from nltk.tokenize import sent_tokenize,word_tokenize,wordpunct_tokenize, TreebankWordTokenizer

corpus = """
Learning is a continuous journey that grows stronger with curiosity and practice. 
Every small effort compound's over time, turning confusion into clarity and ideas into skills. 
By staying consistent, open-minded, and willing to learn from mistakes, 
anyone can steadily move closer to mastery and confidence in their work.
"""

sentence_tokens = sent_tokenize(corpus)
word_tokens = word_tokenize(corpus)
word_punct_tokens = wordpunct_tokenize(corpus)
tree_band_tokenizer = TreebankWordTokenizer()
tree_band_tokens = tree_band_tokenizer.tokenize(corpus)
print("Sentence Tokens:")
for i in range(len(sentence_tokens)):
    print(f"Sentence Tokens in Sentence {i+1}:", (sentence_tokens[i]))
print("Word Tokens:")
for i in range(len(word_tokens)):
    print(f"Character Tokens in Word {i+1} ('{word_tokens[i]}'):")
#Word-Punctuation Tokenization divides words and punctuation into separate tokens like compounds's  to compounds ,' and s.
print("Word-Punctuation Tokens:")
for i in range(len(word_punct_tokens)):
    print(f"Character Tokens in Word {i+1} ('{word_punct_tokens[i]}'):")
# Treebank Word Tokenization handles contractions and punctuation in a more sophisticated manner. like all full stops come with the word except the last one
print("Treebank Word Tokens:")
for i in range(len(tree_band_tokens)):
    print(f"Character Tokens in Word {i+1} ('{tree_band_tokens[i]}'):")