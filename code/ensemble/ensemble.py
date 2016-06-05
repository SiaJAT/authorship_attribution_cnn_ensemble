

import operator
import sys
import os
import pickle
import numpy.matlib
import numpy as np

prior_doc_dist = {}
train_authors = None
total_num_docs = 0
pan_test_labels = {}

pan_original_train = '/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-training-corpus-2012-03-28'
pan_test_labels_path = '/mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-test-corpus-2012-05-24/labels.txt'

pan_test_labels = {}

# get the pat test labels from the test file path and read into global directory
def read_pan_test_labels(test_labels_path):
    global pan_test_labels

    # labels file format
    # doc name (w/o suffix), author as capital letter
    with open(test_labels_path, 'r') as open_file:
        for line in open_file:
            #print line
            entry = line.split(',') 
            file_name, author = entry[0], entry[1].strip()
            pan_test_labels[file_name] = ord(author) - 65


# calculate prior distribution of authors over the train set
def get_author_labels(train_dir):
    train_set = set()
    test_set = set()
    
    train_files = [x for x in sorted(os.listdir(train_dir)) if x != "12Esample01.txt"
            and x != "12Fsample01.txt"
            and x != "README.txt"]

    global prior_doc_dist
    # calculate the prior distribution of the authors
    for file_name in train_files:
        label = file_name.split('train')[1][0]
        train_set.add(label)
        
        label_num = ord(label) - 65

        if label not in prior_doc_dist:
            prior_doc_dist[label_num] = 1
        else:
            prior_doc_dist[label_num] += 1

    global total_num_docs
    for author, freq in prior_doc_dist.iteritems():
        total_num_docs += freq

    global train_authors
    train_authors = train_set

    # get the author test labels
    read_pan_test_labels(pan_test_labels_path)

'''
use a modified voting scheme to determine if
sentence was correctly attributed to author

return 1 if the prediction is correct
return 0 otherwise
'''
def determine_vote(doc_scores_path, output_name):
    file_name = doc_scores_path.split('/')[-1]
    
    # NAMING CONVENTION: "12Atest01_scores.p"
    file_name_no_suffix = file_name.split('_')[0]

    sent_scores_list = pickle.load(open(doc_scores_path, 'rb'))
    
    num_sent, author_dim = sent_scores_list.shape

    total_log_probs = np.zeros(len(train_authors))

    # scale the probabilities by their frequency in the corpus
    for curr_sent in range(num_sent):
        for author in range(len(train_authors)):
            sent_scores_list[curr_sent, author] *= (1.0*prior_doc_dist[author])/total_num_docs

        total_log_probs += sent_scores_list[curr_sent, :] 
    
    # normalize the score by the total sum over outputs
    total_log_probs /= sum(total_log_probs)

    # get the most probable of the classes 
    most_prob_ind = np.where(total_log_probs==max(total_log_probs))
    #most_probable = total_log_probs[most_prob_ind]
    
    with open(output_name + '.txt', 'a') as write_file:
        write_file.write(file_name_no_suffix + ',' + str(most_prob_ind[0][0]) + ',' + str(pan_test_labels[file_name_no_suffix]) + '\n')

    print file_name_no_suffix + ", most probable: " + str(total_log_probs) + ", most probable in: " + str(most_prob_ind[0][0]) + ", real prediction: " + str(pan_test_labels[file_name_no_suffix])

    if most_prob_ind[0][0] == pan_test_labels[file_name_no_suffix]:
        return 1

    return 0


def determine_vote_binary(doc_scores_path, output_name):
    file_name = doc_scores_path.split('/')[-1]
    
    # NAMING CONVENTION: e.g., "12Atest01_0_scores.p"
    file_name_no_suffix = file_name.split('_')[0]

    which_net = int(file_name.split('_')[1])

    sent_scores_list = pickle.load(open(doc_scores_path, 'rb'))
    
    num_sent, author_dim = sent_scores_list.shape

    total_log_probs = np.zeros(2)

    # scale the probabilities by their frequency in the corpus
    for curr_sent in range(num_sent):
        sent_scores_list[curr_sent, 0] *= (1.0*prior_doc_dist[which_net])/total_num_docs
        sent_scores_list[curr_sent, 1] *= (1.0*(total_num_docs -prior_doc_dist[which_net]))/total_num_docs

        total_log_probs += sent_scores_list[curr_sent, :] 
    
    # normalize the score by the total sum over outputs
    total_log_probs /= sum(total_log_probs)

    # get the most probable of the classes 
    most_prob_ind = np.where(total_log_probs==max(total_log_probs))
    #most_probable = total_log_probs[most_prob_ind]
    
    with open(output_name + '.txt', 'a') as write_file:
        write_file.write(file_name_no_suffix + ',' + str(most_prob_ind[0][0]) + ',' + str(pan_test_labels[file_name_no_suffix]) + '\n')

    '''
    print "most probable: " + str(total_log_probs)
    print "most probable in: " + str(most_prob_ind[0][0])
    print "real prediction: " + str(pan_test_labels[file_name_no_suffix])
    '''

    #print file_name_no_suffix + ", most probable: " + str(total_log_probs) + ", most probable in: " + str(most_prob_ind[0][0]) + ", real prediction: " + str(pan_test_labels[file_name_no_suffix])

    print file_name_no_suffix + "," + str(total_log_probs[0])

    if most_prob_ind[0][0] == 0 and which_net == pan_test_labels[file_name_no_suffix]:
        return 1

    if most_prob_ind[0][0] == 1 and which_net != pan_test_labels[file_name_no_suffix]:
        return 1

    return 0


def calc_statistics(test_dir, result_file):
    acc = 0
    total_sent = 0
    
    name2numsent = {}

    with open(os.getcwd() + '/' + result_file, 'r') as read_file:
        for line in read_file:
            file_name, _, file_acc = line.split(',')
            num_sentences, _, _ = pickle.load(open(test_dir + '/' + file_name + '_tensor.p', 'rb')).shape
            name2numsent[file_name] = float(num_sentences) 
            #print type(name2numsent[file_name])
            total_sent += num_sentences
    
    with open(os.getcwd() + '/' + result_file, 'r') as read_file:
        for line in read_file:
            file_name, _, file_acc = line.split(',')

            print "num sent: " + str(name2numsent[file_name]) + ", file acc: " + str(file_acc)
            acc += name2numsent[file_name]*float(file_acc)

    print "sentence level accuracy: " + str(acc/total_sent)

# determine vote for a collection of score documents and get classification accuracy
def determine_test_accuracy(doc_scores_dir, output_name):
    get_author_labels(pan_original_train)
    
    doc_scores_files = os.listdir(os.getcwd() + '/' + doc_scores_dir)

    correct = 0
    total = 0

    for doc in doc_scores_files:
        correct += determine_vote(doc_scores_dir + '/' + doc, output_name)
        total += 1
   
    acc = (1.0*correct) / (1.0*total)
    
    print str(correct)
    print str(acc) 
    return acc 


# determine vote for a collection of score documents and get classification accuracy
def determine_test_accuracy_binary(doc_scores_dir, output_name):
    get_author_labels(pan_original_train)
    
    doc_scores_files = sorted(os.listdir(os.getcwd() + '/' + doc_scores_dir))

    correct = 0
    total = 0

    for doc in doc_scores_files:
        correct += determine_vote_binary(doc_scores_dir + '/' + doc, output_name)
        total += 1
   
    acc = (1.0*correct) / (1.0*total)
    
    #print str(correct)
    #print str(acc) 
    return acc 

def determine_global_accuracy_binary(prediction_index_file):
    get_author_labels(pan_original_train)
     
    doc2scores = {}
    files2open = []
    with open(prediction_index_file, 'r') as pred_file:
        for line in pred_file:
            files2open.append(line.strip())

    files2open = sorted(files2open)
    for file in files2open:
        net_index = int(file.split('_')[1].split('.txt')[0])

        with open(file, 'r') as curr_predict:
            for line in curr_predict:
                file_name, score = line.split(',')
                if file_name not in doc2scores:
                    curr = [0] * 14
                    curr[net_index] = float(score)
                    doc2scores[file_name] = curr
                else:
                    curr = doc2scores[file_name]
                    curr[net_index] = float(score)
                    doc2scores[file_name] = curr

    correct = 0
    total = 0
    for file, scores in doc2scores.iteritems():
        prediction = scores.index(max(scores))
        print "pred: " + str(prediction) + ", actual: " + str(pan_test_labels[file]) +  ", scores: " + str(scores)
        if prediction == pan_test_labels[file]:
            correct += 1
        total += 1

    print "acc: " + str((1.0*correct)/(1.0*total)) + ", correct: " + str(correct) + ", total: " + str(total)


def convert_numeric(file, output):
    with open(file, 'r') as curr_read:
        with open(output, 'w') as results_file:
            for line in curr_read:
               
                file_name, pred, actual = line.split(',')
                pred_num = ord(pred) - 65
                actual_num = ord(actual.strip()) - 65
                results_file.write(file_name + ',' + str(pred_num) + ',' + str(actual_num) + '\n')

def evaluate_ignore(prediction_ref, ignore):
    correct = 0
    total = 0
    with open(prediction_ref, 'r') as pred_file:
        for classif_results_file in pred_file:
            with open(classif_results_file.strip(), 'r') as curr_read:
                for line in curr_read:
                    file_name, pred, actual = line.split(',')
                    pred_num = 1*float(pred)
                    actual_num = 1*float(actual.strip())
                    if file_name != ignore:
                        if pred_num == actual_num:
                            correct += 1
                        total += 1
            
            print classif_results_file.strip() + ", " + str((1.0*correct)/(1.0*total))
            correct = 0
            total = 0

# predition_ref contains names of output_file of each of the classifiers
def get_vote(prediction_ref, ensemble_results_path):
    get_author_labels(pan_original_train)
    
    doc2votes = {}
    doc2truth = {}
    with open(prediction_ref, 'r') as pred_file:
        for classif_results_file in pred_file:
            with open(classif_results_file.strip(), 'r') as curr_classif_pred:
                for line in curr_classif_pred:
                    file_name, pred, actual = line.split(',')
                    if file_name not in doc2votes:
                        doc2votes[file_name] = [1*float(pred)]
                        doc2truth[file_name] = 1*float(actual)
                    else:
                        old_list = doc2votes[file_name]
                        old_list.append(1*float(pred))
                        doc2votes[file_name] = old_list
    
    correct = 0
    total = 0
   
    with open(ensemble_results_path, 'w') as ensemble_results:
        for doc, votes in doc2votes.iteritems():
            most_freq = max(set(votes), key=votes.count)
            print str(doc) + ", " + str(votes) + ", " + str(most_freq)
            if doc != "12Ctest04":
                if most_freq == doc2truth[doc]:
                    correct += 1
                
                ensemble_results.write(doc + ',' + str(most_freq) + ',' + str(doc2truth[doc]) + '\n')
                total +=1

        ensemble_results.write("Correct: " + str(correct) + '\n')
        ensemble_results.write("Total: " + str(total) + '\n')
        ensemble_results.write("Accuracy: " + str((1.0*correct)/(1.0*total)))

if __name__ == "__main__":
    '''
    See README.txt for usage
    '''
    #determine_test_accuracy(sys.argv[1], sys.argv[2])
    #determine_test_accuracy_binary(sys.argv[1], sys.argv[2])
    #determine_global_accuracy_binary(sys.argv[1])
    #get_vote(sys.argv[1], sys.argv[2])
    #evaluate_ignore(sys.argv[1], sys.argv[2])
