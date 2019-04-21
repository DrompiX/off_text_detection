import os
import pickle
import numpy as np
from sklearn.svm import SVC
from utils import tokenize, preprocess

# vocab = []
    # return sorted(list(set(words)))


def make_bow(texts, vocab, save_path=None):
    if save_path is None or not os.path.isfile(save_path):
        vectors = np.zeros((len(texts), len(vocab)))
        for t_id, text in enumerate(texts):
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


def train_bow_classifier(corpus, vocab, labels, model, paths):
    # create_vocabulary(corpus)
    X = make_bow(corpus, vocab, paths['BOW'])
    print(f'X created, {X.shape}')
    Y = np.array(labels)
    print(f'Y created, {Y.shape}')
    # svm = SVC(C=1.0)
    print('Training model...')
    model.fit(X, Y)
    print('Model was trained!')
    # Y_pred = svm.predict(X[4:])
    return model


def predict_with_bow(samples, vocab, model):
    X_test = make_bow(samples, vocab)
    # print(X_test)
    return model.predict(X_test)

# print(classify_bow(["my name is Dima", "hello, World!", "Mafia shit world 228", "hello, my dear slut shit kelly", "U are slut and shit"],
#                    np.array([0, 0, 1, 1])))