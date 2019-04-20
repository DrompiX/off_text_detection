import nltk
import numpy as np
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

def tokenize(text):
    return nltk.word_tokenize(text)

def preprocess(text, remove_stop=True):
    tokenized = tokenize(text.lower())
    if remove_stop:
        return [w for w in tokenized if w.isalpha() and w not in stop_words]
    else:
        return [w for w in tokenized if w.isalpha()]

def create_vocabulary(corpus):
    words = []
    for text in corpus:  
        words.extend(preprocess(text))
    return sorted(list(set(words)))

def make_bow(corpus):
    vocab = create_vocabulary(corpus)
    vectors = np.zeros((len(corpus), len(vocab)))
    for t_id, text in enumerate(corpus):
        words = preprocess(text)
        bag_vector = np.zeros(len(vocab))
        for w in words:
            for i, word in enumerate(vocab):
                if w == word:
                    bag_vector[i] += 1
        vectors[t_id] = bag_vector
    
    return vectors

def classify_bow(corpus):
    pass