#!/usr/bin/python

"""
This script crawls through the ngram
results per author per test document
and picks a label per test document
based on the author LM that
gave it the lowest perplexity. 

The accuracy of the model is then
measured by counting the total correct
and dividing by the total number of tests. 
"""

import math

#Scores a hypothesis vector given a weight vector. 
def score(measures, w):
    return w[0] * -measures[1] + w[1] * measures[2] + w[2] * measures[3]

#Updates the hypothesis vector. 
def update_measures(measures, candidate, w):
    if measures == None:
        return candidate

    if score(candidate, w) < score(measures, w):
        return candidate
    else:
        return measures

#Runs test with given set of weights. 
def test(w):
    labels = {}
    correct_labels = {}

    test_file = open("tests.txt", "r")

    overall_results = open("ngram_results.txt", "w")

    for test in test_file:
        results_file = open(test.strip() + ".results", "r")
        authors_file = open("../authors.txt", "r")
   
        measures = None

        for line in results_file:
            if line.split(" ")[1].strip(",").startswith("zeroprobs"):
                author = authors_file.readline().strip()
                log_prob = float(line.split(" ")[3])
                ppl1 = float(line.split(" ")[5])
                ppl2 = float(line.split(" ")[7])

                candidate = [author, log_prob, ppl1, ppl2]

                measures = update_measures(measures, candidate, w)
       
        #Adds voted label for test file. 
        labels[test.strip()] = measures[0]    

    test_file.close()

    correct_file = open("labels.txt", "r")

    for line in correct_file:
        doc, label = line.split(',')
        correct_labels[doc] = label.strip()

    #Gets correctness measures. 
    total_tests = 0.0
    total_correct = 0.0

    for doc in correct_labels:
        if labels[doc] == correct_labels[doc]:
            total_correct += 1.0

        total_tests += 1.0

        #Writes result to file. 
        overall_results.write(doc + ',' + labels[doc] + ',' + correct_labels[doc] + '\n')

    overall_results.close()

    return total_correct / total_tests

if __name__ == "__main__": 
    w = [0, 0, 1]
                
    result = test(w)

    print result
    
