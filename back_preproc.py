import csv
import os
import sys

def read_and_preproc_data(path):
    
    with open(path) as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        next(rows, None)
        for row in rows:
            ind = row[0]
            label = 0 if row[5] == '2' else 1
            tweet = row[6]
            print(ind, tweet, label)
            sys.exit()


read_and_preproc_data('data.nosync/background_corp.csv')