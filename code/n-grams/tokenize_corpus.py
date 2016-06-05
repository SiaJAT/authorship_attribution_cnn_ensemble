#!/usr/bin/python

"""
This script was used in order to 
tokenize the corpus for use in 
the n-gram experiments. 
"""

import nltk
import sys
import os

for file_name in os.listdir("."):
    if file_name.startswith("12") and file_name.endswith(".txt"):
        pre_file = open(file_name, "r")

        doc = ""

        for line in pre_file:
            doc += line

        pre_file.close()

        print "Tokenizing " + file_name
  
        doc = doc.replace('\n', ' ')
        sentences = nltk.tokenize.sent_tokenize(doc)

        file_name = file_name.split(".")[0]

        post_file = open(file_name + ".tokenized", "w")

        for sent in sentences:
            post_file.write("<s> " + sent + " </s>\n")

        post_file.close()
