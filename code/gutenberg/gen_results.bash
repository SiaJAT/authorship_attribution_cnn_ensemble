#!/bin/bash

#This script calculates perplexity scores for each test document for each author. 

cd mega_test
FILE_NAMES=$( ls *.tok )
cd ..

cd mega_train
LMS=$( ls *.lm )
cd ..

for file_name in $FILE_NAMES; do
	mkdir -p results/$file_name

	echo $file_name

	for lm in $LMS; do
		../SRILM/bin/i686-m64/ngram -lm mega_train/$lm -ppl mega_test/$file_name > results/$file_name/${lm}.txt
	done								   
done
