#!/bin/bash

for i in $( cat authors.txt ); do
	echo $i.txt
	ngram-count -order 2 -kndiscount -interpolate -text train/$i.txt -lm train/$i.lm
done
