import numpy as np
from sql import NLP_Database
import pickle
from gensim import models

class DocumentTensor:
    def __init__(self, name):
        #File prefix name before extension. 
        self.name = name

        #Dimmensions for tensor. 
        self.num_sent = None
        self.max_sent_len = None
        self.word_dimmension = None

        #The tensor. 
        self.tensor = None

        # the database of word vectors
        self.db = None
    
    def build(self, serialized_inverted_index_name, vec_repr):
        if vec_repr == "GLOVE": 
            # instantiate the NLP Dabatase 
            self.db = NLP_Database()

            # set the table
            self.db.pick_table(vec_repr)
        else:
            model = models.Word2Vec.load_word2vec_format('/mnt0/siajat/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
            

        #Load pickled inverted index object.. 
        inverted_index = pickle.load(open(serialized_inverted_index_name, 'rb'))

        #Gets document inverted index dictionary as well as list of statistics for document. 
        word2sent, stats = inverted_index.get_doc_data(self.name)
        
        # WARING: hardcoded word dimension
        self.num_sent = int(stats[0])
        self.max_sent_len = int(stats[1])
        self.word_dimmension = 300

        print "num sent: " + str(self.num_sent)
        print "max sent len: " + str(self.max_sent_len)

        #Gets dimmensions for tensor and initializes it. 
        #TODO Get dimmensions from stats (also implement stats)
        '''
        change max_sent_len param in self.tensor init to HARDCODED VALUE of LIMIT
        '''
        self.tensor = np.zeros((int(self.num_sent), 6, int(self.word_dimmension)))

        #Builds tensor by iterating over every word and every sentence. 
        
        for word in word2sent:
            #print word
            sent2word = word2sent[word]

            for sent in sent2word:
                #Get list of sentence indeces where word appears. 
                word_index_list = sent2word[sent]

                #Place word vector where necessary in sentence matrix. 
                for word_index in word_index_list:
                    if vec_repr == "GLOVE":
                        self.tensor[sent, word_index, :] = self.db.get_wordvec(word) 
                    elif vec_repr == "WORD2VEC":
                        if word in model:
                            self.tensor[sent, word_index, :] = model[word]
                        else:
                            # WARNING: case where entry doesn't exist HARCDOED
                            self.tensor[sent, word_index, :] = np.random.normal(0, 0.1, 300)

    def serialize_tensor(self):
       pickle.dump(self.tensor, open(self.name + "_tensor.p",'wb' )) 
