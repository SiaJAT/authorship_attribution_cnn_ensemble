#!/bin/bash
#This script uses an author pcfg in order to parse
#and score a test document. 

COUNTER=0

for author in $( echo N ); do
	for model in $( ls ../corpus/$author/*pcfg | xargs -n1 basename ); do
		for doc in $( ls ../test ); do
			COUNTER=$((COUNTER + 1))
			echo $COUNTER
			condor_submit -a "TEST_DOC = ../test/$doc" -a "PARSER = ../corpus/$author/$model" -a "RESULTS = ../corpus/$author/test/${model}_${doc}.results" -a "LOGS = ../corpus/$author/test/${model}_${doc}" evaluate_pcfg.condor
		done
	done
done

