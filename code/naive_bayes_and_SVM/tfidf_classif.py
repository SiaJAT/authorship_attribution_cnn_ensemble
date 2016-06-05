import numpy as np
import operator
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, preprocessing
import os
import sys
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

vectorizer = None
X = None
idf = None
idf_dict = None
author_set = ()
pan_test_labels = {}
pan_test_labels_path = '/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-test-corpus-2012-05-24/labels.txt'


def build_corpus(train_path, test_path):
    train_files = [x for x in os.listdir(train_path) if x not in 'README.txt' and 'sample' not in x]
    test_files = [x for x in os.listdir(test_path) if x not in 'README.txt' and x != 'labels.txt' and "12E" not in x and "12F" not in x]
    
    corpus = []
    for f in train_files:
        with open(train_path + '/' + f, 'r') as curr_file:
            corpus.append(unicode(curr_file.read()))
    
    for f in test_files:
        with open(test_path + '/' + f, 'r') as curr_file:
            corpus.append(unicode(curr_file.read()))
    

    global vectorizer
    global X
    global idf
    global idf_dict
    vectorizer = TfidfVectorizer(min_df = 1)
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    idf_dict = dict(zip(vectorizer.get_feature_names(), idf))


def build_tfidf_vec(doc_path):
    tf_dict = {}
    # count the tokens in the document
    with open(doc_path, 'r') as curr_doc:
        all_tokens = word_tokenize(curr_doc.read())
        for token in all_tokens:
            if token not in tf_dict:
                tf_dict[unicode(token)] = 1
            else:
                tf_dict[unicode(token)] += 1
    
    # build the vector
    vec = np.zeros(len(idf_dict.items()))
     
    index = 0
    for term, idf in idf_dict.iteritems():
        #print type(term)
        if term in tf_dict:
            vec[index] = tf_dict[unicode(term)] * idf_dict[unicode(term)]
        else:
            vec[index] = 0
        index += 1

    return vec


def batch_build_tfidf_vecs_classify(train_path, test_path, classifier_type):    
    train_files = sorted([x for x in os.listdir(train_path) if x not in 'README.txt' and 'sample' not in x])
    test_files = sorted([x for x in os.listdir(test_path) if x not in 'README.txt' and x != 'labels.txt' and "12E" not in x and "12F" not in x])
    
    vec_dim = len(idf_dict.items())

    # initialize the training and testing matrices
    train_data_mat = np.zeros((len(train_files), vec_dim))
    test_data_mat = np.zeros((len(test_files), vec_dim))

    # build author label set
    build_author_labels(train_path, test_path)

    # create author labels
    train_labels = get_labels(train_path, 'train')
    test_labels = get_labels(test_path, 'test')

    # read in the the training data
    index = 0
    for f in train_files:
        train_data_mat[index, :] = build_tfidf_vec(train_path + '/' + f)
        index += 1

    index = 0
    for f in test_files:
        test_data_mat[index, :] = build_tfidf_vec(test_path + '/' + f)
        index += 1

    # normalize the data
    train_data_mat = preprocessing.normalize(train_data_mat, norm='l2')
    test_data_mat = preprocessing.normalize(test_data_mat, norm='l2')

    # fit the SVM..... this might be wrong
    #classif_1vr = svm.SVC(kernel='linear', C=100.0)
    if classifier_type == "SVM": 
        classif_1vr = svm.SVC(kernel='linear')
        #classif_1vr = svm.SVC(kernel='rbf', gamma=1.0, degree=2)
        classif_1vr.fit(train_data_mat, train_labels)
        # make prediction
        predict = classif_1vr.predict(test_data_mat) 
    
    if classifier_type == "NB":
        gnb = MultinomialNB()
        predict = gnb.fit(train_data_mat, train_labels).predict(test_data_mat)
        print str(gnb.class_log_prior_)

    # make prediction
    #predict = classif_1vr.predict(test_data_mat)
    
    # write all the predictions to a file
    # format = file_name, prediction, actual
    with open(classifier_type + '_results.txt', 'wb') as write_file:
        for i in xrange(0, len(test_files)):
            curr_file_str  = test_files[i]
            curr_file_no_suffix = curr_file_str.split('.txt')[0]
            write_file.write(curr_file_no_suffix + ', ' + str(predict[i]) + ', ' + str(test_labels[i]) + '\n')
        
    print "predictions: " + str(predict)
    print "test labels: " + str(test_labels)
    print "train labels: " + str(train_labels)

    # calculate the accuracy
    acc = calc_acc(test_labels, predict)
    
    return acc


def read_pan_test_labels(test_labels_path):
    global pan_test_labels

    with open(test_labels_path, 'r') as open_file:
        for line in open_file:
            entry = line.split(',')
            file_name, author = entry[0], entry[1].strip()
            pan_test_labels[file_name] = author


def build_author_labels(train_dir, test_dir):
    train_set = set()
    test_set = set()

    for file_name in [x for x in os.listdir(train_dir) if x not in 'README.txt' and 'sample' not in x]:
        #print file_name
        train_set.add(file_name.split('train')[1][0])
    
    read_pan_test_labels(pan_test_labels_path)
    for test_file, test_author in pan_test_labels.iteritems():
        test_set.add(test_author)


    global author_set
    author_set = train_set

def get_labels(dir_path, train_or_test):
    labe_vector = None
    if train_or_test == 'train':
        train_files = sorted([x for x in os.listdir(dir_path) if x != 'README.txt' and 'sample' not in x]) 
        label_vector = np.zeros(len(train_files))
        
        index = 0
        for f in train_files:
            author_index = ord(f.split('n')[1][0]) - 65 
            label_vector[index] = author_index
            index += 1

    else:
        test_files = sorted([x for x in os.listdir(dir_path) if x != 'labels.txt' and "12E" not in x and "12F" not in x and "README" not in x])
        label_vector = np.zeros(len(test_files))

        index = 0
        for f in test_files:
            file_name_no_suffix = f.split('.')[0]
            label_vector[index] = ord(pan_test_labels[file_name_no_suffix]) - 65
            index += 1
    
    return label_vector

def calc_acc(targets, predict):
    num_correct = sum([1 for i in xrange(0, len(targets)) if targets[i] == predict[i]])
    return num_correct*1.0 / len(targets)*1.0


if __name__ == "__main__":
    build_corpus(sys.argv[1], sys.argv[2])
    #vec = build_tfidf_vec(sys.argv[3])
    acc = batch_build_tfidf_vecs_classify(sys.argv[1], sys.argv[2], sys.argv[3])
    print str(acc)
