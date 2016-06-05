#!/bin/bash
#This file submits all pcfg generation jobs to condor. 

for i in $( echo C E I B ); do
	condor_submit -a "AUTHOR_PATH = ../corpus/$i/" -a "LOGS = ../corpus/$i/pcfg" gen_pcfgs.condor
done
