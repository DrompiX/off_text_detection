import numpy as np
from hateword2vec import HateWord2Vec
from hatedoc2vec import HateDoc2Vec
from bow_classifier import BOWClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class Hate2Vec(object):

    def __init__(self, vocab, paths):
        self.paths = paths
        self.vocab = vocab
    
    def fit(self, X, Y, dirty_list):
        print('> Fitting Hate2Vec model...')
        print('>> Fitting HateWord2Vec model...')
        self.hw2v = HateWord2Vec(self.paths['background'], self.paths['w2v'])
        self.hw2v.fit(dirty_list, self.vocab, t=0.955, w=10)
        print('>> HateWord2Vec was fitted.')
        print('>> Fitting HateDoc2Vec model...')
        log_clf = LogisticRegression(C=10, class_weight='balanced', solver='saga', 
                                     n_jobs=-1, penalty='l1', max_iter=1000)
        self.hd2v = HateDoc2Vec(X, self.paths['d2v'])
        self.hd2v.fit(log_clf, Y)
        print('>> HateDoc2Vec was fitted.')
        print('>> Fitting BOWClassifier...')
        rf_clf = RandomForestClassifier(n_estimators=25, class_weight='balanced', 
                                        n_jobs=-1, min_samples_leaf=3)
        self.bow2clf = BOWClassifier(X, rf_clf)
        self.bow2clf.fit(X, Y, self.paths['BOW'])
        print('>> BOWClassifier was fitted.')

        hw2v_labels_pred = self.hw2v.predict(X)
        hd2v_labels_pred = self.hd2v.predict(X)
        bow2clf_labels_pred = self.bow2clf.predict(X)

        print('>> Fitting meta classifier...')
        X_meta = np.column_stack((hw2v_labels_pred, hd2v_labels_pred, bow2clf_labels_pred))
        self.meta_clf = RandomForestClassifier(n_estimators=15)
        self.meta_clf.fit(X_meta, Y)
        print('>> Meta classifier was fitted.')
        print('> Hate2Vec model was fitted!')

    def predict(self, X):
        hw2v_labels_pred = self.hw2v.predict(X)
        hd2v_labels_pred = self.hd2v.predict(X)
        bow2clf_labels_pred = self.bow2clf.predict(X)
        X_meta = np.column_stack((hw2v_labels_pred, hd2v_labels_pred, bow2clf_labels_pred))
        return self.meta_clf.predict(X_meta)
