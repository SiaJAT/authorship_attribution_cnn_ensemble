#!/bin/bash
#This script submits all tree-generation (for each author) jobs to condor. 

for author in $( echo A B C D E F G H I J K L M N ); do
	condor_submit gen_trees.condor -a "LOGS = ../corpus/$author/trees" -a "SENTENCES = ../corpus/$author/tokenized.txt" -a "MEGA_CORPUS = ../mega_corpus.mrg" -a "TREES = ../corpus/$author/trees.txt" 
done
