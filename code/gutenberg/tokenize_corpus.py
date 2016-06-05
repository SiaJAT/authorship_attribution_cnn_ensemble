#!/usr/bin/python

"""
This script was used to tokenize the corpus. 
"""

import os
import sys
import nltk

#Tokenizes each file in a directory. 
def tokenize_dir(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            print file_name
            tokenize_file(directory, file_name)

#Tokenizes a file. 
def tokenize_file(directory, file_name):
    doc_string = ""
    doc = open(directory + '/' + file_name, 'r')
    tokenized_doc = open(directory + '/' + file_name + '.tok', 'w')

    for line in doc:
        doc_string += line.replace('\n', ' ')

    sentences = nltk.tokenize.sent_tokenize(doc_string)

    for sent in sentences:
        tokenized_doc.write(sent + '\n')

    doc.close()
    tokenized_doc.close()

if __name__ == "__main__":
    print "TRAIN"
    tokenize_dir("mega_train")

    print "TEST"
    tokenize_dir("mega_test")
