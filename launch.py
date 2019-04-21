import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
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
    print(sum(np.array(labels) == 0), sum(np.array(labels) == 1))
    # dirty_list = read_dirty_list(paths['dirty'])
    # hw2v = HateWord2Vec(paths['background'], paths['w2v'])
    # hw2v.train(dirty_list, vocab, t=0.955, w=10)
    # # hw2v.save_model(paths['w2v'])
    # labels_pred = hw2v.predict(corpus)
    # print(np.sum(np.array(labels_pred) == np.array(labels)) / len(labels))
    # print(roc_auc_score(np.array(labels), np.array(labels_pred)))
    # print(f1_score(np.array(labels), np.array(labels_pred)))

    hd2v = HateDodc2Vec(corpus)
    hd2v.train(LogisticRegression(C=10, class_weight='balanced', solver='saga', n_jobs=-1), labels)
    print(hd2v.predict(["you bitches love yall some corny nigga"]))
    hd2v_labels_pred = hd2v.predict(corpus)
    print(np.sum(np.array(hd2v_labels_pred) == np.array(labels)) / len(labels))
    # print(hd2v.model.infer_vector(preprocess("you bitches love yall some corny nigga")))
    
    # print(len(hd2v.model.docvecs.vectors_docs))

    # model = RandomForestClassifier(10)
    # model = train_bow_classifier(corpus, labels, model, paths)
    # print(predict_with_bow(np.array(["hey, you are slut and bitch!"]), model))
    # print(predict_with_bow(np.array(["Thank you! You are really kind!"]), model))


if __name__ == '__main__':
    launch()
