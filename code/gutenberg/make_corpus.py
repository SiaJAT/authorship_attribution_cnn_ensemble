#!/usr/bin/python

"""
This script 
"""

import os
import sys

#Appends one file to another. 
def append_file(src_name, dst_name):
    src = open(src_name, 'r')
    dst = open(dst_name, 'a')

    for line in src:
        dst.write(line)

#Makes one full text file for each author in the corpus. 
for file_name in os.listdir("mega_train"):
    if file_name.endswith(".tok"):
        author = file_name.split("___")[0]

        #Appends file to full author file. 
        append_file("mega_train/" + file_name, "mega_train/" + author + ".full")

