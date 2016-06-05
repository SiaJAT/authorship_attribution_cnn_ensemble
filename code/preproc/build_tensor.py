#!/usr/bin/python

import numpy as np
import sys
import os
import subprocess as sp
from document_tensor import DocumentTensor

def batch_build_pan_train(output_dir, dir, inv_ind_file, model_type):
    '''
    files = [x.split('.txt')[0] for x in os.listdir(dir) 
            if x != "12Esample01.txt" and 
            x != "12Fsample01.txt" and
            x != "README.txt" and 
            "12E" not in x and
            "12F" not in x
            and "12" in x]
    '''

    files = [x.split('.txt')[0] for x in os.listdir(dir)]
    for f in files:
        tens = DocumentTensor(f)
        tens.build(inv_ind_file, model_type)
        tens.serialize_tensor()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for f in files:
        file_name = f + '.p'
        sp.Popen(['mv', file_name, output_dir])

# nohup python build_tensor.py pan_lim_train /mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-training-corpus-2012-03-28 pan_train.p WORD2VEC &> pan_training.log & 
if __name__ == "__main__": 
    batch_build_pan_train(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
