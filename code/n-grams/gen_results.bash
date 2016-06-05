#!/bin/bash

#This script generates result files for each author and test document. 

for test in $( cat test/tests.txt ); do
	echo $test    

	for author in $( cat authors.txt ); do
        ngram -lm train/$author.lm -ppl test/$test.tokenized >> test/$test.results
    done
done
