import csv
import os
import sys

# def read_and_preproc_data(path):
#     corpus = []
#     with open('data.nosync/background_corp.csv', 'w+') as csvout:
#         writer = csv.writer(csvout)
#         with open(path) as csvfile:
#             rows = csv.reader(csvfile, delimiter=',')
#             next(rows, None)
#             for row in rows:
#                 # ind = row[0]
#                 # label = 0 if row[5] == '2' else 1
#                 tweet = row[6].replace('\n', ' ')
#                 writer.writerow([tweet])

# read_and_preproc_data('data.nosync/background.csv')