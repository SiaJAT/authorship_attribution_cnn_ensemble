#!/usr/bin/python

"""
This script gathers statistics for
authors in the gutenberg corpus. 
"""

import os
import sys

author_stats = {}

for file_name in os.listdir("mega_train"):
    if file_name.endswith(".tok"):
        author = file_name.split('___')[0]

        print author

        if author not in author_stats:
            author_stats[author] = 0.0

        author_stats[author] += 1.0

minimum = float('inf')
maximum = float('-inf')

for author in author_stats:
    if author_stats[author] > maximum:
        maximum = author_stats[author]

    if author_stats[author] < minimum and author_stats[author] > 1: 
        minimum = author_stats[author]

print "MAX: " + str(maximum)
print "MIN: " + str(minimum)
