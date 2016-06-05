#!/usr/bin/python

"""
To train each author pcfg, we used
the WSJ and Brown corpora along with
appended and up-sampled treebanks from 
each author. This script was used in order
to determine the up-sampling proportions
for each author. 
"""

#Appends one file to the other. 
def append_file(src, dst):
    src_file = open(src, 'r')
    dst_file = open(dst, 'a')

    dst_file.write('\n')

    for line in src_file:
        dst_file.write(line)

    src_file.close()
    dst_file.close()

counts_file = open("counts.txt", "r")
counts = {}

#Gets tree counts for each author. 
for line in counts_file:
    count, author = line.split()

    counts[author] = float(count)

mega_corpus_count = 63992.0

#Builds different proportioned treebanks per author.
#In the end, we only used the highest percent proportion
#for each author. 
for author in counts:
    proportions = {}

    total = mega_corpus_count + counts[author]
    initial_proportion = counts[author] / total

    proportion = initial_proportion
    done = False
    fifty_fifty = False
    multiplier = 1

    #Sets initial proportion without multiplying. 
    proportions[str(multiplier)] = proportion

    while not done:
        if proportion >= 0.5 and proportion < 0.75:
            if not fifty_fifty: 
                proportions[str(multiplier)] = proportion
                fifty_fifty = True
        elif proportion >= 0.75:
            proportions[str(multiplier)] = proportion
            
            done = True
     
        multiplier += 1
        total = (counts[author] * multiplier) + mega_corpus_count 
        proportion = (counts[author] * multiplier) / total 

    #Creates treebank file for other proportions.        
    proportion_file = open(author + "/proportions.txt", "w")

    for proportion in proportions:
        proportion_file.write(str(proportions[proportion]) + " " + str(proportion) + "\n")

    proportion_file.close()
