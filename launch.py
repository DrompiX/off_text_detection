import csv
import numpy as np
from bow_classifier import train_bow_classifier, predict_with_bow
from sklearn.ensemble import RandomForestClassifier


def read_data(path):
    ids, corpus, labels = [], [], []
    with open(path) as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        for t_id, t_text, t_class in rows:
            ids.append(t_id)
            corpus.append(t_text)
            labels.append(0 if t_class == 'none' else 1)
    
    return ids, corpus, labels
            

def launch():
    paths = {
        "tweets": 'data.nosync/tweets.csv',
        "BOW": 'data.nosync/bow_tweets.p',
        "background": 'data.nosync/background_corp.csv'
    }
    ids, corpus, labels = read_data(paths['tweets'])
    print(sum(np.array(labels) == 0), sum(np.array(labels) == 1))
    # model = RandomForestClassifier(10)
    # model = train_bow_classifier(corpus, labels, model, paths)
    # print(predict_with_bow(np.array(["hey, you are slut and bitch!"]), model))
    # print(predict_with_bow(np.array(["Thank you! You are really kind!"]), model))


if __name__ == '__main__':
    launch()
