import nltk
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