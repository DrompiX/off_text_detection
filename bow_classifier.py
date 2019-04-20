import nltk

def tokenize(text):
    return nltk.word_tokenize(text)

def preprocess(text):
    tokenized = tokenize(text.lower())
    return [w for w in tokenized if w.isalpha()]

def create_vocabulary(corpus):
    words = []
    for text in corpus:
        words.extend(preprocess(text))
    
    return sorted(list(set(words)))

def make_BOW(corpus):
    vocab = create_vocabulary(corpus)
    pass

def classify_BOW(corpus):
    pass