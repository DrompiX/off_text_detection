import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from bow_classifier import train_bow_classifier, predict_with_bow
from hateword2vec import HateWord2Vec
from hatedoc2vec import HateDodc2Vec
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


def launch():
    paths = {
        "tweets": 'data.nosync/tweets.csv',
        "BOW": 'data.nosync/bow_tweets.p',
        "background": 'data.nosync/background_corp.csv',
        "dirty": 'data.nosync/dirty_list.txt',
        "dirty_learned": 'data.nosync/dirty_learned.txt',
        "w2v": 'data.nosync/word2vec.model'
    }
    ids, corpus, labels = read_data(paths['tweets'])
    vocab = create_vocabulary(corpus)
    # print(sum(np.array(labels) == 0), sum(np.array(labels) == 1))
    # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=9)
    x_train, x_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.25, 
                                                        stratify=labels, random_state=51)

    dirty_list = read_dirty_list(paths['dirty'])
    hw2v = HateWord2Vec(paths['background'], paths['w2v'])
    hw2v.train(dirty_list, vocab, t=0.955, w=10)
    # hw2v.save_model(paths['w2v'])
    hw2v_labels_pred = hw2v.predict(x_test)
    print(np.sum(np.array(hw2v_labels_pred) == np.array(y_test)) / len(y_test))
    # print(roc_auc_score(np.array(labels), np.array(hw2v_labels_pred)))
    # print(f1_score(np.array(labels), np.array(hw2v_labels_pred)))

    hd2v = HateDodc2Vec(x_train)
    hd2v.train(LogisticRegression(C=10, class_weight='balanced', solver='saga', n_jobs=-1), y_train)
    # print(hd2v.predict(["you bitches love yall some corny nigga"]))
    hd2v_labels_pred = hd2v.predict(x_test)
    print(np.sum(np.array(hd2v_labels_pred) == np.array(y_test)) / len(y_test))
    # print(hd2v.model.infer_vector(preprocess("you bitches love yall some corny nigga")))
    
    # print(len(hd2v.model.docvecs.vectors_docs))

    rf_model = RandomForestClassifier(n_estimators=25, class_weight='balanced', n_jobs=-1, verbose=1, min_samples_leaf=1)
    rf_model = train_bow_classifier(x_train, vocab, y_train, rf_model, paths)
    # print(predict_with_bow(np.array(["hey, you are slut and bitch!"]), vocab, rf_model))
    # print(predict_with_bow(np.array(["Thank you! You are really kind!"]), vocab, rf_model))
    rf_labels_pred = predict_with_bow(x_test, vocab, rf_model)
    print(np.sum(np.array(rf_labels_pred) == np.array(y_test)) / len(y_test))

    # good_prob = sum(np.array(labels) == 0) / len(labels)
    # bad_prob = 1 - good_prob
    meta_clf = MultinomialNB()#class_prior=[good_prob, bad_prob])
    X_meta = np.column_stack((hw2v_labels_pred, hd2v_labels_pred, rf_labels_pred))
    meta_clf.fit(X_meta, y_test)
    meta_labels_pred = meta_clf.predict(X_meta)
    print(np.sum(np.array(meta_labels_pred) == np.array(y_test)) / len(y_test))
    

if __name__ == '__main__':
    launch()
