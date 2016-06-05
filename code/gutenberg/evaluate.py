#!/usr/bin/python

"""
This script was used in order
to evaluate the results of the ngram
model performance on our gutenberg corpus. 

The perplexity values were first recorded in 
individual files, so this script crawls through
them to calculate the best scoring author in order
to pick a label per test set document. 

Accuracy is then measured by dividing the number
of correct labels by the number of total tests. 
"""

import os
import sys

w = [0.0, 0.0, 1.0]

def score(candidate):
    return candidate[1] * w[0] + candidate[2] * w[1] + candidate[3] * w[2]

def update_class(candidate, current):
    if score(candidate) < score(current):
        return candidate
    else:
        return current

total_tests = 0.0
total_correct = 0.0

for directory in os.listdir("results"):
    correct_label = directory.split("___")[0]

    current_best = None
    best_ppl = float('inf')

    #Gets results from each candidate author's LM
    for result_file_name in os.listdir("results/" + directory):
        candidate_name = result_file_name.split(".full")[0]

        result_file = open("results/" + directory + "/" + result_file_name, 'r')

        for line in result_file:
            if line.split(' ')[1].startswith("zeroprobs"):
                results = line.split(' ')
                
                log_prob = float(results[3])
                ppl1 = float(results[5])
                ppl2 = float(results[7])

                candidate = [candidate_name, log_prob, ppl1, ppl2]
    
                if ppl2 < best_ppl:
                    current_best = candidate_name
                    best_ppl = ppl2

                #current_best = update_class(candidate, current_best)
 
    print current_best + ',' + correct_label

    if correct_label == current_best:
        total_correct += 1.0

    total_tests += 1.0

acc = total_correct / total_tests

print acc
