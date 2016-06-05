#!/bin/usr/python

"""
This file crawls through 
all of the result files from 
the author parser scoring for each
test file. It then uses these scores
to determine a label for each test. 
The chosen labels are compared to the correct
labels from the test set and the model's
performance is evaluated by measuring the
accuracy in this regard. 
"""

import os
import sys

doc_results = {}

#Gets results for each author for each test document. 
for author in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]:
    path = "corpus/" + author + "/test/"

    for result_file_name in os.listdir(path):
        result_file = open(result_file_name, 'r')
        score = float(result_file.readline().strip())

        test_name = result_file.split(".pcfg_")[1].split(".")[0]

        if test_name not in doc_results:
            doc_results[test_name] = {}

        doc_results[test_name][author] = score

#Gets correct labels. 
correct_labels = {}

label_file = open("raw_test/labels.txt", 'r')

for line in label_file:
    doc, label = line.split(',')
    label = label.strip()

    correct_labels[doc] = label

#Gets accuracy. 
total_correct = 0.0
total_tests = float(len(doc_results))

for test in doc_results:
    best = None
    best_score = float('-inf')

    for author in doc_results[test]:
        if doc_results[test][author] > best_score:
            best = author
            best_score = doc_results[test][author]

    if best == correct_labels[test]:
        total_correct += 1.0

acc = total_correct / total_tests

print acc
