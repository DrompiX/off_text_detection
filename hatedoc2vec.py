import csv
import os
import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from utils import preprocess

class HateDoc2Vec(object):

    def __init__(self, corpus, model_path=None):
        if model_path is None:
            if corpus is not None:
                self.model = self._build_model(corpus)
            else:
                raise RuntimeError("Please, specify path to corpus")
        else:
            print('Loading model...')
            self.model = Doc2Vec.load(model_path)
            print('Model was loaded!')
        
    def fit(self, clf, labels):
        self.clf = clf
        X = self.model.docvecs.vectors_docs
        Y = labels 
        clf.fit(X, Y)
    
    def predict(self, texts):
        result = []
        for text in texts:
            cleaned_text = preprocess(text)
            text_vector = self.model.infer_vector(cleaned_text)
            result.append(self.clf.predict([text_vector])[0])
        return result
    
    def _build_model(self, corpus):
        data = [TaggedDocument(preprocess(t), [i]) for i, t in enumerate(corpus)]
        model = Doc2Vec(data, vector_size=200, window=10, min_count=3, workers=4)
        print('Start training model...')
        model.train(data, total_examples=len(data), epochs=15)
        print('Model training finished!')
        return model
    
    def save_model(self, path):
        print("Saving model...")
        self.model.save(path)
        print('Model was saved!')