#!/bin/bash

#This script uses SRILM in order to make a language model for 
#each author. 

for author_file in $( ls mega_train/*.full ); do
	echo $author_file
	../SRILM/bin/i686-m64/ngram-count -kndiscount -interpolate -text $author_file -lm ${author_file}.lm
done
