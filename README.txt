Authorship Attribution Ensemble Using CNNs
Rodolfo Corona (UTEID: rc38783)
Julian Sia (UTEID: jas7794)



REMARK:
Some of these code files were adapted to fit this problem, namely:
1. tfidf_classif.py (modfied implementation of SVM classification based on assignment written for CS 391L by Julian Sia, extended and modified to handle Naive Bayes and tf-idf vectorized input)
2. Parser388.java (modified CS 388 HW3 code for Rodolfo Corona)

Code Contents (code)
1. cnn
	- rhodes_cnn.py = code to train and test the CNN neural nets
	- rhodes_cnn_binary.py = code to train and test binary neural nets for 1 vs. rest classification
2. preproc
	- inverted_index.py = a simple inverted index for a corpus  
	- build_inverted_index.py = given a training or testing set, creates an inverted index of documents (used in transforming documents into tensors)
	- document_tensor.py = a document tensor object (wraps a multidimensional numpy array and document statistics)
	- build_tensor.py = a call to this builds the document tensors given an inverted index and a corpus of documents
	- sql.py = used to create a sql database of GloVE vectors originally stored as a txt file
3. ensemble
	- ensemble.py = methods for ensembling predictions from different classifiers into one judgment
4. naive_bayes_and_SVM
- tfidf_classif.py = tfidf classification using Naive Bayes or SVM (used for trainining naive bayes and SVM classifiers in ensemble)
5. gutenberg
	 - (see BELOW)
6. pcfg
	- (see BELOW)
7. gutenberg
	- (see BELOW)

Supporting Files (supporting_txt)
1. ensemble_members.txt = list of ensemble members to be used in conjunction with 'get_vote' in 'ensemble.py'
2. ensemble_nocnn.txt =  list of ensemble members (excluding CNNs) to be used in conjunction with 'get_vote' in 'ensemble.py'

Results File (results)
1. SVM_results.txt = prediction results for SVM classification
2. NB_results.txt = prediction results for Naive Bayes classification
3. CNN_GLOVE_baseline.txt = prediction results for CNN trained of GloVe vectors
4. CNN_WORD2VEC_baseline.txt = prediction results for CNN trainined on word2vec vectors
5. pcfg_results_numeric.txt = prediction results for PCFG
6. ngrams_results_numeric.txt = prediction results for ngram language model classification
7. ensemble_results_nocnn.txt = prediction results for ensemble without CNNs
8. ensemble_results.txt = prediction results for ensemble with CNNs


Datasets (data)
1. pan12-authorship-attribution-training-corpus-2012-03-28 = a cleaned version of the training corpus we used, with encoding errors fixed
2. pan12-authorship-attribution-test-corpus-2012-05-24 = a cleaned version of the testing corpus we used, with encoding errors fixed (see 'labels.txt' for test files that actually have an author)
3. gutenberg - the gutenberg corpus used to collect n-gram information (UNATTACHED: available at 
4. GloVE Vectors - for vectorizing sentences (UNATTACHED: available at http://nlp.stanford.edu/data/glove.6B.zip)
5. word2vec Vectors - for vectorizing sentences (UNATTACHED: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)


Dependenceis for Python Code
1. tensorflow
2. keras
3. scikit-learn
4. numpy
5. matplotlib
6. mysql

Remarks on Usage:

For training and testing CNNs
1. rhodes_cnn.py 
	- (USAGE: python rhodes_cnn.py <WORD2VEC|GLOVE> <output_name> )
	- uncomment out 'train_routine' or 'test_routine' to train or test using parameter settings
	- parameter settings were modified throughout project
	- general architectur ecan be found in 'build_rhodes()'
	- see keras documentation for more details (http://keras.io/)
	- evaluation done using 'ensemble.py'

2. rhodes_cnn_binary.py
	- (USAGE: python rhodes_cnn_binar.py)
	- uncomment out first or second with block to train or test, respectively
	- produce train and test output for 14 binary classification neural networks
	- evaluation done using 'ensemble.py'

3. ensemble.py
	UNCOMMENT out the following to run
 	- get_vote 
 		* (USAGE: python ensemble.py <name of ensemble members file in current dir> <output results file to current directory>)
 		* does plurality voting given a list containing names of prediction files output by each classifier and write output to a given output file
 	- evaluate_ignore
 		* (USAGE: python ensemble.py <name of ensemble member file in current dir> <ignore file>)
 		* Print out the classification scores of each ensemble member, ignoring the ignore file when calculating accuracy
 	- determine_test_accuracy
 		* (USAGE: python ensemble.py <directory containin score files for each test document> <file to write predictions>)
 		*  Given a directory containing score pickles for each test document, write a prediction file where each line consists of 
 			a file name, prediction, and true label in comma separated format (eg: 12test04, 1, 2)
 	- determine_test_accuracy_binary
 		* identical usage as 'determine_test_accuracy', but for output of the binary neural networks
 	- determine_global_accuracy_binary
 		* (USAGE: python ensemble.py <file containing names of binary neural net classifier predictions)
 		* Given a file containing names of 14 binary neural net classifiers, assign to a document the label of the most confident binary classifier 

 4. python tfidf_classif.py
 	* (USAGE: python tfidf_classif.py <training text files directory> <testing text files directory> <SVM|NB>)
 	* Trains and tests either an SVM or NB (third argument specifies which) and outputs classification results to 'SVM_results.txt' for SVM and 'NB_results.txt' for Naive Bayes

 5. python build_inverted_index.py
 	* (USAGE eg:  python build_inverted_index.py /mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-training-corpus-2012-03-28 pan_train)
 	* Given a directory to a training or testing documents and an output name, build and serialize an inverted index writing a pickle file of the index
 	* ASSUMES dependency 'inverted_index.py' is in directory

 6. python build_tensor.py
 	* (USAGE eg:  python build_tensor.py pan_lim_train /mnt0/siajat/cs388/nlp/data/pan12-authorship-attribution-training-corpus-2012-03-28 pan_train.p)
 	* Given a directory to a set of training or testing documents and an inverted index pickle file, create document tensors with dimensions (num sentences, 6, 300) for each document in training or testing set and serialize tensors to current directory
 	* ASSUMES dependency 'document_tensor.py' and 'sql.py'


---------
gutenberg
---------

	-----------
	evaluate.py
	-----------
	This script was used
	in order to evaluate the accuracy
	of the ngram model on the gutenberg
	corpus. 

	----------------
	gen_results.bash
	----------------
	This script calculates perplexity
	scores for each author language model and
	test document using the SRILM toolkit.

	------------
	get_stats.py
	------------
	This script gathers statistics 
	such (e.g. number of documents) for
	each author in the gutenberg corpus. 


	--------------
	make_corpus.py
	--------------
	This file takes every tokenized 
	document for each author and appends them
	into one file per author. This was done
	in order to create the author language models.

	-------------
	make_lms.bash
	-------------
	This script takes each authors full
	tokenized file and makes a language model
	for them using the SRILM toolkit.

	------------------
	tokenize_corpus.py
	------------------
	This script tokenizes the 
	train and test sets so that
	language models may be created
	properly per author. 

------
n-gram
------

	-----------
	evaluate.py
	-----------
	This script is used to evaluate the 
	results of the n-gram experiments on 
	the test set. 


	------------
	gen_lms.bash
	------------
	This script was used 
	to generate all of the author
	LMs that were used for the ngram
	experiments. 

	----------------
	gen_results.bash
	----------------
	This script was used to generate
	the result files from the experiments.
	The result files contain the log probability
	and perplexity that the given LM assigns to the
	given document. 
	The result files are then fed to evaluate.py
	in order to measure the performance of the model. 

	------------------
	tokenize_corpus.py
	------------------
	This script was used in order to tokenize the corpus
	for use with the SRILM binaries which both generated
	ngram LMs and then evaluated their perplexity on test
	set documents. 

--------
pcfg
--------
	---------------
	Parser388.java
	---------------
	This file contains code that
	was used to interface with the 
	Stanford Parser. The program
	uses a command line interface
	in order to specify the action. 
	It was used in order to take a 
	tokenized text file for each author 
	in order to parse it and create a treebank,
	load the treebank and train a pcfg on it, 
	and using it to score the likelihood that 
	the author wrote a particular document.

	------------------
	get_proportions.py
	------------------
	This file was used to determine the upsampling
	proportions for each authors treebank in order 
	to be able to append it to the WSJ and Brown treebank
	that was created for training. 

	------------------
	tokenize_corpus.py
	------------------
	This file was used to tokenize the 
	train and test sets so that they could
	be parsed properly by the Stanford
	Parser. 

	------------
	get_stats.py
	------------
	This file was used in order to 
	determine the proportion that
	each author made up in the training
	corpus. This statistic was gathered
	for error analysis purposes. 

	-----------
	evaluate.py
	-----------
	This script was used to evaluate the
	results of the pcfg experiments. 

	---------------------
	submit_pcfg_jobs.bash
	---------------------
	This script was used to submit
	all pcfg generating jobs to condor. 

	---------------------
	submit_tree_jobs.bash
	---------------------
	This script was used to submit
	all treebank generating jobs to
	condor. 

	---------------------------
	submit_evaluation_jobs.bash
	---------------------------
	This script was used in order
	to submit all test document
	evaluation jobs for the author 
	pcfgs. 

	--------------------
	gen_trees.condor
	gen_pcfgs.condor
	evaluate_pcfg.condor
	--------------------
	These were the condor submit
	files that were used for all 
	the pcfg related jobs. 
