import os
import pickle
import numpy as np
from utils import preprocess


class BOWClassifier(object):

    def __init__(self, vocab, clf):
        self.vocab = vocab
        self.clf = clf

    def _make_bow(self, texts, save_path=None):
        if save_path is None or not os.path.isfile(save_path):
            vectors = np.zeros((len(texts), len(self.vocab)))
            for t_id, text in enumerate(texts):
                words = preprocess(text)
                bag_vector = np.zeros(len(self.vocab))
                for w in words:
                    for i, word in enumerate(self.vocab):
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
    
    def fit(self, corpus, labels, path):
        X = self._make_bow(corpus, path)
        print(f'X created, {X.shape}')
        Y = np.array(labels)
        print(f'Y created, {Y.shape}')
        print('Training model...')
        self.clf.fit(X, Y)
        print('Model was trained!')
    
    def predict(self, samples):
        X_test = self._make_bow(samples)
        return self.clf.predict(X_test)

# def make_bow(texts, vocab, save_path=None):
#     if save_path is None or not os.path.isfile(save_path):
#         vectors = np.zeros((len(texts), len(vocab)))
#         for t_id, text in enumerate(texts):
#             words = preprocess(text)
#             bag_vector = np.zeros(len(vocab))
#             for w in words:
#                 for i, word in enumerate(vocab):
#                     if w == word:
#                         bag_vector[i] += 1
#             vectors[t_id] = bag_vector
        
#         if save_path is not None:
#             print('Saving bow vectors.')
#             with open(save_path, 'wb+') as fd:
#                 pickle.dump(vectors, fd)

#     elif save_path is not None:
#         print('Loading bow vectors.')
#         with open(save_path, 'rb') as fd:
#             vectors = pickle.load(fd)
    
#     return vectors


# def train_bow_classifier(corpus, vocab, labels, model, paths):
#     X = make_bow(corpus, vocab, paths['BOW'])
#     print(f'X created, {X.shape}')
#     Y = np.array(labels)
#     print(f'Y created, {Y.shape}')
#     print('Training model...')
#     model.fit(X, Y)
#     print('Model was trained!')
#     return model


# def predict_with_bow(samples, vocab, model):
#     X_test = make_bow(samples, vocab)
#     return model.predict(X_test)
