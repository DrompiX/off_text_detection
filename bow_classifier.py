import os
import pickle
import numpy as np
from utils import preprocess
from gensim import corpora


class BOWClassifier(object):

    def __init__(self, corpus, clf):
        preproc_texts = [preprocess(text) for text in corpus]
        self.dictionary = corpora.Dictionary(preproc_texts)
        self.clf = clf

    def _make_bow(self, texts, save_path=None):
        if save_path is None or not os.path.isfile(save_path):
            vectors = np.zeros((len(texts), len(self.dictionary.token2id.keys())))
            for i, text in enumerate(texts):
                bow = self.dictionary.doc2bow(preprocess(text))
                for k, v in bow:
                    vectors[i][k] = v
            
            if save_path is not None:
                print('Saving bow vectors.')
                with open(save_path, 'wb') as fd:
                    pickle.dump(vectors, fd)

        elif save_path is not None:
            print('Loading bow vectors.')
            with open(save_path, 'rb') as fd:
                vectors = pickle.load(fd)
        
        return vectors
    
    def fit(self, corpus, labels, path=None):
        X = self._make_bow(corpus, path)
        Y = np.array(labels)
        print('Training model...')
        self.clf.fit(X, Y)
        print('Model was trained!')
    
    def predict(self, samples):
        X_test = self._make_bow(samples)
        return self.clf.predict(X_test)
