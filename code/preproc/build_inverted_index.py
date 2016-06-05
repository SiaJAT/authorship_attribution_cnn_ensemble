#!/usr/bin/python

import sys
import numpy
import sql
from inverted_index import InvertedIndex

# USAAGE:  nohup python build_inverted_index.py /mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-training-corpus-2012-03-28 pan_train &> pan_build_train.log &
if __name__ == "__main__":
    #Construct inverted index. 
    index = InvertedIndex()
    #index.build_representation("/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-test-corpus-2012-05-24")
    #index.serialize_inverted_index("pan_test")

    index.build_representation(sys.argv[1])
    index.serialize_inverted_index(sys.argv[2])
