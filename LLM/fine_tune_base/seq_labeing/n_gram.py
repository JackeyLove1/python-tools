from nltk import bigrams, trigrams
from collections import Counter, defaultdict
data_text = """ Hello \n World!"""
n_gram_sents = [i.strip().split(' ') for i in data_text.strip().split('\n') if i]

model = defaultdict(lambda :defaultdict(lambda : 0))
list(trigrams())
