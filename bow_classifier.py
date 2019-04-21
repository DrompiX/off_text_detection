import os
import nltk
import pickle
import numpy as np
from nltk.corpus import stopwords
from sklearn.svm import SVC

stop_words = stopwords.words('english')

vocab = []

def tokenize(text):
    return nltk.word_tokenize(text)

def preprocess(text, remove_stop=True):
    tokenized = tokenize(text.lower())
    if remove_stop:
        return [w for w in tokenized if w.isalpha() and w not in stop_words]
    else:
        return [w for w in tokenized if w.isalpha()]

def create_vocabulary(corpus):
    global vocab
    words = []
    for text in corpus:  
        words.extend(preprocess(text))
    vocab = sorted(list(set(words)))
    # return sorted(list(set(words)))

def make_bow(corpus, save_path=None):
    global vocab
    if save_path is None or not os.path.isfile(save_path):
        vectors = np.zeros((len(corpus), len(vocab)))
        for t_id, text in enumerate(corpus):
            words = preprocess(text)
            bag_vector = np.zeros(len(vocab))
            for w in words:
                for i, word in enumerate(vocab):
                    if w == word:
                        bag_vector[i] += 1
            vectors[t_id] = bag_vector
        
        if save_path is not None:
            print('Saving bow vectors.')
            with open(save_path, 'wb+') as fd:
                pickle.dump(vectors, fd)

    elif save_path is not None:
        print('Loading bow vectors.')
        with open(save_path, 'rb') as fd:
            vectors = pickle.load(fd)
    
    return vectors

def train_bow_classifier(corpus, labels, model, paths):
    create_vocabulary(corpus)
    X = make_bow(corpus, paths['BOW'])
    print(f'X created, {X.shape}')
    Y = np.array(labels)
    print(f'Y created, {Y.shape}')
    # svm = SVC(C=1.0)
    print('Training model...')
    model.fit(X, Y)
    print('Model was trained!')
    # Y_pred = svm.predict(X[4:])
    return model


def predict_with_bow(samples, model):
    X_test = make_bow(samples)
    # print(X_test)
    return model.predict(X_test)

# print(classify_bow(["my name is Dima", "hello, World!", "Mafia shit world 228", "hello, my dear slut shit kelly", "U are slut and shit"],
#                    np.array([0, 0, 1, 1])))