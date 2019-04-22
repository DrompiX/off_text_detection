import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from hate2vec import Hate2Vec
from utils import preprocess


def read_data(path):
    ids, corpus, labels = [], [], []
    with open(path) as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        for t_id, t_text, t_class in rows:
            ids.append(t_id)
            corpus.append(t_text)
            labels.append(0 if t_class == 'none' else 1)
    
    return ids, corpus, labels


def create_vocabulary(corpus):
    words = []
    for text in corpus:  
        words.extend(preprocess(text))
    return sorted(list(set(words)))


def read_dirty_list(path):
    with open(path, 'r') as fd:
        dirty_list = fd.readlines()
        dirty_list = [w.rstrip() for w in dirty_list]
        return dirty_list


def get_scores(y_true, y_pred):
    print('Overall accuracy:', np.sum(np.array(y_pred) == np.array(y_true)) / len(y_true))
    print('F1-score:', f1_score(y_true, y_pred))
    print('ROC-AUC score:', roc_auc_score(y_true, y_pred))
    print('Total cnt:', len(y_true))
    print('1 cnt:', np.sum(np.array(y_pred) == 1), '| 0 cnt:', np.sum(np.array(y_pred) == 0))


def launch():
    paths = {
        "tweets": 'data.nosync/tweets.csv',
        "BOW": 'data.nosync/bow_tweets.p',
        "background": 'data.nosync/background_corp.csv',
        "background2": 'data.nosync/dataset.csv',
        "dirty": 'data.nosync/dirty_list.txt',
        "dirty_learned": 'data.nosync/dirty_learned.txt',
        "w2v": 'data.nosync/word2vec.model',
        "d2v": 'data.nosync/doc2vec.model'
    }
    ids, corpus, labels = read_data(paths['tweets'])
    vocab = create_vocabulary(corpus)
    x_train, x_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.25, 
                                                        stratify=labels, random_state=51)

    dirty_list = read_dirty_list(paths['dirty'])

    h2v = Hate2Vec(vocab, paths)
    h2v.fit(x_train, y_train, dirty_list)
    h2v_labels_pred = h2v.predict(x_test)
    get_scores(y_test, h2v_labels_pred)

    # while True:
    #     text = input('write some text: ')
    #     print('offensive' if h2v.predict([text])[0] == 1 else 'not offensive')


if __name__ == '__main__':
    launch()
