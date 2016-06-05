import sys
import os
import subprocess as sp
import pprint
import random
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pickle

class InvertedIndex:
    
    '''
    an inverted index with the following structure:
    
    {word : {doc : [(s_1, [w_1, ..., w_n])]}}
    
    a dictionary from words to a list of tuples, 
    where each tuple maps from a sentence index 
    '''
    def __init__(self):
        self.doc2word = {}
        self.doc2stats = {}
                

    def insert(self, word, doc, sent, word_pos):
        # this document has been seen before
        if doc in self.doc2word:
            word2sent = self.doc2word[doc]
            
            # this document containing this word has been seen before
            if word in word2sent:
                sent2word_ind = word2sent[word]
                
                # this document containing this word has seen this sentence before and this word is in this sentence 
                if sent in sent2word_ind:
                    word_indices = sent2word_ind[sent]
                    word_indices.append(word_pos)
                    sent2word_ind[sent] = word_indices
                
                # this word in this document has been seen in a different sentence
                else:
                    sent2word_ind[sent] = [word_pos]

            # this word has not been seen in this document before
            else:
                word2sent[word] = {sent : [word_pos]}
            
        # first time the document  has been seen in the corpus
        else:
            sent2word_ind = {sent: [word_pos]}
            word2sent = {word: sent2word_ind}
            self.doc2word[doc] = word2sent
            
    
    def build_representation(self, curr_dir):
        
        reload(sys)
        sys.setdefaultencoding('utf8')

        # list all of the words in the directory 
        curr_dir_files = [x for x in sorted(os.listdir(curr_dir)) if x is not "12Esample01.txt" 
                and x is not "12Fsample01.txt" 
                and x is not "README.txt"]

        #print curr_dir_files

        # serialize pretrained docs written to "glove_dict.p"
        # presently hard coded 
        #serialize_pretrained_vecs_data('/mnt0/siajat/cs388/nlp/data/glove_sample.txt', "glove")

        #word_dim = len(word_list)
         
        # save the old directory and go to where the
        # other files are stored
        old_dir = os.getcwd()
        os.chdir(curr_dir)
     

        for f in curr_dir_files:
            doc_name = f.split('.txt')[0]
            
            with open(f, 'r') as curr_file:
                # clean the string
                curr_file_string = curr_file.read().replace('\n', ' ').replace('\r', '')
                #curr_file_string = unicode(curr_file_string, errors='replace')
                
                # segment the sentence collectin stats 
                sent_segm = sent_tokenize(curr_file_string)
                #self.doc2stats[doc_name] = (len(sent_segm),self.len_longest_sentence(curr_file_string))
               
		small_sent_list = []

		for sent in sent_segm:	
		        sent = word_tokenize(sent)
                        #Change values to something else if necessary. 
			sent_size = 6
			allowed_remainder = 4

			if len(sent) > sent_size:
				#Split
				num_bins = len(sent) / sent_size
				remainder = len(sent) % sent_size

				base = 0
				bound = 0

				for i in range(num_bins):
					base = i * sent_size
					bound = (i + 1) * sent_size

					small_sent_list.append(sent[base:bound])

				if remainder >= allowed_remainder:
					small_sent_list.append(sent[bound:])
				

                        if len(sent) >= 4 and len(sent) < sent_size:
				#If fits, then ships.  
                                small_sent_list.append(sent)

                self.doc2stats[doc_name] = (len(small_sent_list), 6)
                # tokenize each sentence
                sent_counter = 0
                for sent in small_sent_list:
                    #tok_segm = word_tokenize(sent)
                    
                    tok_counter = 0 
                    for tok in sent:   
                        self.insert(tok, doc_name, sent_counter, tok_counter)
                        tok_counter += 1    
                    
                    sent_counter += 1
                
                print "doc: " + doc_name + ", sentence count: " + str(sent_counter)
        # change back to the original working directory
        os.chdir(old_dir)    
    
    def len_longest_sentence(self, curr_file_string):
        sent_segm = sent_tokenize(curr_file_string)
    
        longest = 0

        for sent in sent_segm:
            longest = max(len(word_tokenize(sent)), longest)

        return longest

    
    def serialize_inverted_index(self, save_name):
        pickle.dump(self, open(save_name + ".p", 'wb'))

    
    def get_doc_data(self, doc_name):
        word2sent = self.doc2word[doc_name]  
        stats = self.doc2stats[doc_name] 
        return (word2sent, stats)

def serialize_pretrained_vecs_data(train_path, vec_type):
    file_len = 0
    with open(train_path, 'r') as read_file:
        for line in read_file:
            file_len += 1

    label2vec = {}

    with open(train_path, 'r') as read_file:
        for line in read_file:
            arr = line.split(' ') 

            curr_label = arr[0]
            curr_vec = arr[1:] 
            
            label2vec[curr_label] = np.array([float(x.strip()) for x in curr_vec])

    pickle.dump(label2vec, open(vec_type + "_dict.p", "wb"))
