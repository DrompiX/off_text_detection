import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from hate2vec import Hate2Vec
from utils import preprocess

# TODO: remove
from bow_classifier import BOWClassifier
from sklearn.ensemble import RandomForestClassifier
from hateword2vec import HateWord2Vec
from hatedoc2vec import HateDoc2Vec
from sklearn.linear_model import LogisticRegression


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
    global vocab
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
    print(np.sum(np.array(y_pred) == np.array(y_true)) / len(y_true))
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
    
    # TODO: tune first two models

    # rf_clf = RandomForestClassifier(n_estimators=25, class_weight='balanced', 
    #                                     n_jobs=-1, min_samples_leaf=3)
    # bow2clf = BOWClassifier(vocab, rf_clf)
    # bow2clf.fit(x_train, y_train, paths['BOW'])
    # bow2clf_labels_pred = bow2clf.predict(x_test)
    # get_scores(y_test, bow2clf_labels_pred)

    # hw2v = HateWord2Vec([paths['background'], paths['background2']], paths['w2v'])
    # hw2v.fit(dirty_list, vocab, t=0.85, w=3) #0.85 3(4)
    # hw2v_labels_pred = hw2v.predict(x_test)
    # get_scores(y_test, hw2v_labels_pred)

    # log_clf = LogisticRegression(C=10, class_weight='balanced', solver='saga', 
    #                              n_jobs=-1, penalty='l1', max_iter=1000)
    # hd2v = HateDoc2Vec(x_train, paths['d2v'])
    # hd2v.fit(log_clf, y_train)
    # hd2v_labels_pred = hd2v.predict(x_test)
    # get_scores(y_test, hd2v_labels_pred)
    # while True:
    #     text = input('write some text: ')
    #     print('offensive' if hw2v.predict([text])[0] == 1 else 'not offensive')

    # TODO: write function to poll for queries to be classified
    

if __name__ == '__main__':
    launch()
