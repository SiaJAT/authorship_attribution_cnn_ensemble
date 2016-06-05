#!/usr/bin/python

"""
This file was used to tokenize all the files in a folder
(i.e. the train and test folders) so that they files could
be parsed for training author treebanks. 
"""

import os
import nltk
import sys


folder = sys.argv[1]

#Usage: ./tokenize_corpus.py [corpus/test]

#Tokenizes corpus. 
for file_name in os.listdir(folder):
        #Tokenizes the author's file. 
        full_file = open(folder + file_name, "r")
        tokenized_file = open(folder + file_name + ".tokenized", "w")

        doc = ""

        for line in full_file:
            doc += line

        full_file.close()

        sentences = nltk.tokenize.sent_tokenize(doc)

        for sent in sentences:
            sent_list = nltk.tokenize.word_tokenize(sent)
            sent = ""

            for word in sent_list:
                sent += word + ' ' 

            sent = sent.strip() + '\n'

            tokenized_file.write(sent)
