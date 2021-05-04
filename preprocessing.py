# libraries
import numpy as np
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from collections import Counter # Count most common words

class Preprocess():
    def __init__(self, book_id):
        self.lang = 'English'
        self.path = '/Users/chris/Documents/GitHub/IRTM/'
        self.root = ET.fromstring(open(self.path + self.lang + '.xml').read())
        self.book_id = book_id

    def tokenize(self):
        tokens = []
        with open(self.lang + '-' + self.book_id + '.txt', 'w', encoding='utf-8') as out:
            for n in self.root.findall('.//div[@id="'+self.book_id+'"]/*seg'):
                tokens += nltk.tokenize.word_tokenize(n.text.strip() + '\n')
        return tokens

    def frequent_words(self, tokens):
        print(Counter(tokens))

    def remove_stopwords(self, tokens):
        return [word for word in tokens if not word in stopwords.words()]

    def remove_punctuation(self, tokens):
        return [word for word in tokens if word.isalnum()]

    def remove_all(self,tokens):
        return [word for word in tokens if ((not word in stopwords.words()) and word.isalnum())]

    def pos_tag(self, tokens):
        return nltk.pos_tag(tokens)


book_luke = Preprocess('b.LUK')
tokens = book_luke.tokenize()
tokens = book_luke.remove_all(tokens)
print(len(tokens))
tokens = book_luke.pos_tag(tokens)
print(tokens)

