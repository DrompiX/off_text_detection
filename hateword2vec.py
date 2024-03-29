import csv
import os
import gensim
from utils import tokenize, preprocess

class HateWord2Vec(object):

    def __init__(self, path_to_corp, model_path=None):
        if model_path is None:
            if path_to_corp is not None:
                self.model = self._build_model(path_to_corp)
            else:
                raise RuntimeError("Please, specify path to corpus")
        else:
            print('Loading model...')
            self.model = gensim.models.Word2Vec.load(model_path)
            print('Model was loaded!')
        
    def fit(self, dirty_list, vocab, t, w):
        """Fit HateWord2Vec model
        
        Arguments:
            texts {list[str]} -- texts to train model om
            t {[type]} -- threshold for similarity between words
            w {[type]} -- threshold for 
        """
        new_dirty_list = dirty_list.copy()
        for word in vocab:
            cur_w = 0
            for dirty in dirty_list:
                if word in self.model.wv.vocab and dirty in self.model.wv.vocab:
                    similarity = self.model.wv.similarity(word, dirty)
                    if similarity > t:
                        cur_w += 1

            if cur_w >= w:
                new_dirty_list.append(word)
        
        self.dirty_list = new_dirty_list
    
    def predict(self, texts):
        results = []
        for text in texts:
            words = set(preprocess(text))
            offensive = False
            for word in words:
                if word in self.dirty_list:
                    offensive = True
                    break

            results.append(1 if offensive else 0)
        
        return results
    
    def _build_model(self, path_to_corp):
        data = self._read_data(path_to_corp[0]) + self._read_data2(path_to_corp[1])
        model = gensim.models.Word2Vec(data, size=100, window=10, min_count=1, workers=4)
        print('Start training model...')
        model.train(data, total_examples=len(data), epochs=15)
        print('Model training finished!')
        return model

    def _read_data(self, path):
        data = []
        with open(path) as fd:
            for line in fd:
                data.append(preprocess(line.rstrip()))
        return data
    
    def _read_data2(self, path):
        data = []
        labels = set()
        with open(path) as csvfile:
            rows = csv.reader(csvfile)
            next(rows, None)
            for label, tweet in rows:
                if label not in labels:
                    labels.add(label)
                
                if label != 'none':
                    data.append(preprocess(tweet))
        print(labels)
        return data
    
    def save_model(self, path):
        print("Saving model...")
        self.model.save(path)
        print('Model was saved!')