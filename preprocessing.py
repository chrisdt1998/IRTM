# libraries
import numpy as np
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from collections import Counter # Count most common words
import pickle

class Preprocess():
    def __init__(self, book_id, output_name):
        self.lang = 'English'
        self.path = '/Users/chris/Documents/GitHub/IRTM/'
        self.root = ET.fromstring(open(self.path + self.lang + '.xml').read())
        self.book_id = book_id
        self.tokens = []
        self.output_name = output_name

    def read_tokenize(self):
        with open(self.lang + '-' + self.book_id + '.txt', 'w', encoding='utf-8') as out:
            for n in self.root.findall('.//div[@id="'+self.book_id+'"]/*seg'):
                self.tokens += nltk.tokenize.word_tokenize(n.text.strip() + '\n')

    def frequent_words(self):
        print(Counter(self.tokens))

    def remove_stopwords(self):
        return [word for word in self.tokens if not word in stopwords.words()]

    def remove_punctuation(self):
        return [word for word in self.tokens if word.isalnum()]

    def remove_all(self):
        return [word for word in self.tokens if ((not word in stopwords.words()) and word.isalnum())]

    def pos_tag(self):
        return nltk.pos_tag(self.tokens)

    def save(self):
        with open(self.output_name + ".pkl", "wb") as out:
            pickle.dump(self.tokens, out)




book_luke = Preprocess('b.LUK', "processed_text")
book_luke.read_tokenize()
book_luke.remove_all()
book_luke.pos_tag()
book_luke.save()



