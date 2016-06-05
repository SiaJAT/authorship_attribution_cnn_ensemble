#!/usr/bin/python

"""
This file was used to collect proportion statistics
per author in the training corpus. 
"""

import os
import sys

author_stats = {}
total = 0.0

for author in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]:
    corpus_file = open("corpus/" + author + "/tokenized.txt", 'r')

    author_stats[author] = 0.0

    for line in corpus_file:
        author_stats[author] += 1

    total += author_stats[author]

for author in author_stats:
    print author + ' : ' + str(author_stats[author] / total) + ' : ' + str(author_stats[author])
