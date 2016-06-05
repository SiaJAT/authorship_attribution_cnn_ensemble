#!/bin/usr/python

import operator
import tensorflow as tf
import sys
import os
import pickle
import numpy.matlib
import numpy as np
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import convolutional, Merge, advanced_activations
from keras.optimizers import Adagrad
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
#import theano
#theano.config.device='gpu0'
#theano.config.floatX = 'float32'

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

GLOVE_train = '/mnt0/siajat/cs388/nlp_project_share/scripts/pan_train_GLOVE'
GLOVE_test = '/mnt0/siajat/cs388/nlp_project_share/scripts/pan_test_GLOVE'

W2V_test_subset = '/mnt0/siajat/cs388/nlp_project_share/scripts/pan_test_WORD2VEC_subset'
W2V_subset = '/mnt0/siajat/cs388/nlp_project_share/scripts/pan_train_WORD2VEC_subset'
W2V_train = '/mnt0/siajat/cs388/nlp_project_share/scripts/pan_train_WORD2VEC'
W2V_test = '/mnt0/siajat/cs388/nlp_project_share/scripts/pan_test_WORD2VEC'
pan_original_train = '/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-training-corpus-2012-03-28'
pan_original_test = '/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-test-corpus-2012-05-24' 
pan_test_labels_path = '/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-test-corpus-2012-05-24/labels.txt'

sample_train = '/mnt0/siajat/cs388/nlp_project_share/scripts/pan_train_WORD2VEC/12AtrainA1_tensor.p'
sample_test = '/mnt0/siajat/cs388/nlp_project_share/scripts/pan_test_WORD2VEC/12Atest01_tensor.p'


longest_sentence_corpus = 6
merged_model = None
author_set = None
pan_test_labels = {}

def center_sentence_tensor(tensor_file):
    curr_doc_tensor = pickle.load(open(tensor_file, 'rb'))
    num_sent, max_sent_len, word_dim = curr_doc_tensor.shape
    
    #print tensor_file
    #print "num sent: " + str(num_sent)
    #print "max sent len: " + str(max_sent_len)
    #print "word dim: " + str(word_dim)
    #print "global max sent: " + str(longest_sentence_corpus)

    # initialize a random tensor 
    centered_tensor = np.random.normal(0, 0.1, (num_sent, longest_sentence_corpus, word_dim))
    #print centered_tensor.shape

    for sent_ind in xrange(num_sent - 1, -1, -1):
        # find the len of this sentence
        word_ind = max_sent_len - 1
        while word_ind >= 0 and sum(curr_doc_tensor[sent_ind, word_ind, :]) == 0:
            word_ind -= 1
        
        # bias towards having no padding instead of having some...
        pad_amount = (longest_sentence_corpus - word_ind)/2
        
        #print "pad: " + str(pad_amount)
        #print "curr_doc_tensor shape: " + str(curr_doc_tensor.shape)

        start_write = pad_amount
        end_write = start_write + word_ind + 1
        start_read = 0
        end_read = word_ind + 1

        centered_tensor[sent_ind, start_write:end_write,:] = curr_doc_tensor[sent_ind, start_read:end_read, :]

         

    centered_tensor = l2normalize(centered_tensor)
    return centered_tensor

def l2normalize(curr_tensor):
    num_sent, sent_len, word_dim = curr_tensor.shape
    for sent_ind in xrange(0, num_sent):
        curr = curr_tensor[sent_ind, :, :]
        curr = curr.flatten()
        curr = np.reshape(curr, (1, len(curr)))
        curr = preprocessing.normalize(curr)
        curr = np.reshape(curr, (sent_len, word_dim))
        curr_tensor[sent_ind, :, :] = curr

    return curr_tensor

def get_longest_corpus_sentence(original_train, original_test): 
    train_files = [x for x in sorted(os.listdir(original_train)) if x != "12Esample01.txt"
            and x != "12Fsample01.txt"
            and x != "README.txt"]
    test_files = [x for x in sorted(os.listdir(original_test)) if x != 'ground-truth.txt'
            and x != 'README.txt']
 
    longest = 0
    for curr_train in train_files:
        #print curr_train
        read_file = open(original_train + '/' + curr_train, 'r')
        curr_file_string = read_file.read().replace('\n', ' ').replace('\r', '')
        curr_file_string = unicode(curr_file_string, errors='replace')
        sent_segm = sent_tokenize(curr_file_string)

        for sent in sent_segm:
            longest = max(len(word_tokenize(sent)), longest)
        read_file.close()

    for curr_test in test_files:
        #print curr_test
        read_file = open(original_test + '/' + curr_test, 'r')
        curr_file_string = read_file.read().replace('\n', ' ').replace('\r', '')
        curr_file_string = unicode(curr_file_string, errors='replace')
        sent_segm = sent_tokenize(curr_file_string)

        for sent in sent_segm:
            longest = max(len(word_tokenize(sent)), longest)
        read_file.close()

    global longest_sentence_corpus
    longest_sentence_corpus = longest
    print "done getting longest"


def build_rhodes():
    print "building rhodes"
    auth_model_3gram = Sequential()
    auth_model_3gram.add(convolutional.Convolution1D(100, 3, border_mode='same', input_shape=(longest_sentence_corpus, 300)))    
    auth_model_3gram.add(convolutional.MaxPooling1D(2, stride=1, border_mode='same')) 

    auth_model_4gram = Sequential()
    auth_model_4gram.add(convolutional.Convolution1D(100, 4, border_mode='same', input_shape=(longest_sentence_corpus, 300)))
    auth_model_4gram.add(convolutional.MaxPooling1D(2, stride=1, border_mode='same'))   

    auth_model_5gram = Sequential()
    auth_model_5gram.add(convolutional.Convolution1D(100, 5, border_mode='same', input_shape=(longest_sentence_corpus, 300)))
    auth_model_5gram.add(convolutional.MaxPooling1D(2, stride=1, border_mode='same'))   

    global merged_model
    merged_model = Sequential()
    merged_model.add(Merge([auth_model_3gram, auth_model_4gram, auth_model_5gram], mode='concat', concat_axis=2))
    merged_model.add(Flatten())
    merged_model.add(Dense(200))
    merged_model.add(Dense(200, activation='relu'))
    merged_model.add(Dropout(0.5))
    merged_model.add(Dense(200))
    merged_model.add(Dense(2, activation='softmax'))
    
    merged_model.summary()
    
    ada = Adagrad(lr=0.0001, epsilon=1e-06)
    merged_model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['accuracy'])
    
    
def get_author_labels(train_dir, test_dir):
    train_set = set()
    test_set = set()

    for file_name in os.listdir(train_dir):
        #print file_name.split('train')
        train_set.add(file_name.split('train')[1][0])

    # get the PAN test labels
    read_pan_test_labels(pan_test_labels_path)
    for test_file, test_author in pan_test_labels.iteritems():
        #print test_file + ", " + test_author
        test_set.add(test_author)
    

    assert len(test_set - train_set) >= 0
    global author_set
    author_set = train_set
    #print str(author_set)

def train_tensor(doc_tensor_path, which_net):
    # 1 hot vector encoding for categorical cross entropy loss (multinomial logistic regression)
    file_name = doc_tensor_path.split('/')[-1]
    
    # create author vector (ASSUME that num author determined: get_author_labels())
    author_index = ord(file_name.split('n')[1][0]) - 65
    label_vector = np.zeros((1,2))

    if author_index == which_net:
        label_vector[0,0] = 1.0
    else:
        label_vector[0,1] = 1.0
    
    doc_tensor = center_sentence_tensor(doc_tensor_path)
    num_sent, max_sent_len, word_dim = doc_tensor.shape
    label_mat = numpy.matlib.repmat(label_vector, num_sent, 1) 
    
    history = LossHistory()
    global merged_model
    
    print "training using: " + file_name
    '''
    for curr_sent_ind in xrange(0, num_sent):
        reshape_tensor = np.zeros((1, max_sent_len, word_dim))
        reshape_tensor[0, :, :] = doc_tensor[curr_sent_ind,:,:] 
        
        history = merged_model.train_on_batch([reshape_tensor, reshape_tensor, reshape_tensor], label_vector)     
    '''
    history = merged_model.train_on_batch([doc_tensor, doc_tensor, doc_tensor], label_mat)
    #history = merged_model.fit([doc_tensor, doc_tensor, doc_tensor], label_mat, batch_size=64, verbose=1)

def batch_train(train_dir, which_net):
    train_pickles = os.listdir(train_dir)
    for f in train_pickles:
        train_tensor(train_dir + '/' + f, which_net)


def test_tensor(doc_tensor_path, corpus_id, passed_model, which_net):
    file_name = doc_tensor_path.split('/')[-1]
    
    file_name_no_suffix = file_name.split('_')[0]
    
    #print "no suffix: " + file_name_no_suffix
    # error: test file has no label
    if "train" not in file_name_no_suffix and corpus_id is "PAN" and file_name_no_suffix not in pan_test_labels:
        print "file not found"
        return 
    
    # get the document tensor
    doc_tensor = center_sentence_tensor(doc_tensor_path)
    num_sent, max_sent_len, word_dim = doc_tensor.shape
    
    # get the author of this document
    if "train" in file_name_no_suffix:
        author_index = ord(file_name.split('n')[1][0]) - 65
    else:
        author_index = pan_test_labels[file_name_no_suffix]
        author_index = ord(author_index.strip()) - 65

    # create author vector (ASSUME that num author determined: get_author_labels())
    
    label_vector = np.zeros((1,2))
    if author_index == which_net:
        label_vector[0,0] = 1.0
    else:
        label_vector[0,1] = 1.0

    # create a label vector for each sentence and store in a matrix
    label_mat = numpy.matlib.repmat(label_vector, num_sent, 1)
    
    #merged_model = model_from_json(open('naive_run.json').read())
    #merged_model.load_weights('naive_run.h5')
    # evaluate
    #print type(merged_model)
    
    score = passed_model.evaluate([doc_tensor, doc_tensor, doc_tensor], label_mat)
    with open('results_1d.txt', 'a') as write_file:
        write_file.write(file_name_no_suffix + "," + str(score[0]) + "," + str(score[1]) + "\n")

    classes = passed_model.predict_classes([doc_tensor, doc_tensor, doc_tensor])
    proba = passed_model.predict_proba([doc_tensor, doc_tensor, doc_tensor])
    #print str(proba.shape)
    #print str(proba[0])
    #print str(type(proba[0][0]))
    #print str(type(proba[0]))
    #print str(type(proba))
    #print str(determine_prediction(classes, author_index))
    pickle.dump(proba, open(file_name_no_suffix + '_' + str(which_net) + '_prob.p','wb'))  
    pickle.dump(classes, open(file_name_no_suffix + '_' + str(which_net) + '_classes.p','wb'))
    #print file_name_no_suffix + ", " + str(classes) + ", " + str(proba)

def batch_test(test_dir, corpus_id, merged_model, which_net):
    test_pickles = os.listdir(test_dir)
    for f in test_pickles:
        test_tensor(test_dir + '/' + f, corpus_id, merged_model, which_net)

def determine_prediction(votes, true_class):
    choice2votes = {}
    for vote in votes:
        if vote not in choice2votes:
            choice2votes[vote] = 1
        else:
            choice2votes[vote] += 1

    winner = -1
    winner_vote_count = -1
    for choice, vote_count in choice2votes.iteritems():
        if vote_count > winner_vote_count:
            winner = choice
            winner_vote_count = vote_count

    return winner == true_class

def read_pan_test_labels(test_labels_path):
    global pan_test_labels

    # labels file format
    # doc name (w/o suffix), author as capital letter
    with open(test_labels_path, 'r') as open_file:
        for line in open_file:
            #print line
            entry = line.split(',') 
            file_name, author = entry[0], entry[1].strip()
            pan_test_labels[file_name] = author


def save_model_disk(save_name):
    global merged_model
    merged_model.save_weights(save_name + '.h5')
    json_string = merged_model.to_json()
    open(save_name + '.json','w').write(json_string)

def make_clean():
    if os.path.isfile('naive_run_1d_GLOVE.h5'):
        os.remove('naive_run_1d_GLOVE.h5')
    if os.path.isfile('naive_run_1d_GLOVE.json'):
        os.remove('naive_run_1d_GLOVE.json')

def load_and_test(test_dir_path, corpus_id, json_file, h5_file, which_net):
    global merged_model
    merged_model = Sequential()
    merged_model = model_from_json(open(json_file).read())
    merged_model.load_weights(h5_file)
    ada = Adagrad(lr=0.0001, epsilon=1e-06)
    merged_model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['accuracy'])
    batch_test(W2V_test, 'PAN', merged_model, which_net)

def load_and_test_and_iterative(test_dir_path, corpus_id, json_file, h5_file, which_net):
    for var in [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]:
        global merged_model
        print "var: " + str(var)
        merged_model = Sequential()
        merged_model = model_from_json(open(json_file).read())
        merged_model.load_weights(h5_file)
        ada = Adagrad(lr=var, epsilon=1e-06)
        merged_model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['accuracy'])
        batch_test(W2V_test, 'PAN', merged_model, which_net)  
    
def train_routine(rep_type, which_net):
    make_clean()
    get_author_labels(W2V_train, W2V_test)
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=tf_config):
    '''
    build_rhodes()
    if rep_type == "WORD2VEC":
        batch_train(W2V_train, which_net)
    else:
        batch_train(GLOVE_train, which_net)
    save_model_disk(rep_type + '_' + str(which_net))

def test_routine(rep_type, which_net):
    get_author_labels(W2V_train, W2V_test)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    model_name = rep_type + '_' + str(which_net)
    with tf.Session(config=tf_config):
        if rep_type == "WORD2VEC":
            load_and_test(W2V_test, 'PAN', os.getcwd() + '/' + model_name + '.json', os.getcwd() + '/' + model_name + '.h5', int(which_net))
        else:
            load_and_test(GLOVE_test, 'PAN', os.getcwd() + '/' + model_name + '.json', os.getcwd() + '/' + model_name + '.h5', int(which_net))

if __name__ == "__main__": 
    '''
    REMARK: uncomment first with block to train 14 binary networks using WORD2VEC word vectors
            uncomment second with block to test 14 binary networks using WORD2VEc word vectors
            evaluation code to produce scores can be found in 'ensemble.py'
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    '''
    with tf.Session(config=tf_config):
        for i in xrange(0,14):
            train_routine("WORD2VEC", i)
    '''
    '''
    with tf.Session(config=tf_config):
        for i in xrange(0,14):
            test_routine("WORD2VEC", i)   
    '''
