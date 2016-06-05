import java.util.Collection;
import java.util.*;
import java.io.*;
import java.lang.Integer;
import java.lang.Math.*; 

import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.Options;
import edu.stanford.nlp.parser.lexparser.EvaluateTreebank; 

public class Parser388 {
	/** Command line interface to run parser experiments. 
	    Options include training a pcfg for an author, 
		parsing a tokenized text file in order to create
		a treebank for an author, and evaluating a test
		document using an author's pcfg in order to get 
		a score for it. 
   	*/
	public static void main(String[] args) {
		//Checks that correct number of arguments is given. 
		if (args.length < 0) {
			System.out.println("No type of action specified!");
			System.exit(1); 
		}

		if (args[0].equals("train")) {
			//Further check for argument numbers. 
			if (args.length != 3) {
				System.out.println("Usage: $java Parser388 train [mega treebank] [author path]"); 
				System.exit(1); 
			}
			
			train(args, false); 
		}
		else if (args[0].equals("parse")) {
			//Further check for argument numbers. 
			if (args.length != 4) {
				System.out.println("Usage: $java Parser388 parse [mega corpus] [raw corpus] [treebank]"); 
				System.exit(1); 
			}	

			parse_corpus(args); 
		}
		else if (args[0].equals("test")) {
			//Ensures correct number of arguments are used. 
			if (args.length != 4) {
				System.out.println("Usage: $java Parser388 test [test document] [parser] [results file]"); 
				System.exit(1); 
			}

			//Sets flags for parser. 
			ArrayList<String> flags = new ArrayList<String>();
			flags.add("-goodPCFG"); 
			flags.add("-evals"); 
			flags.add("tsv");

			//Sets options for treebank. 
			Options op = new Options(); 
			op.doDep = false;
			op.doPCFG = true;
			op.setOptions("-goodPCFG", "-evals", "tsv"); 

			//Loads existing parser from file. 
			LexicalizedParser lp = LexicalizedParser.loadModel(args[2], flags);  

			System.out.println("Successfully loaded parser from file.");

			//Parses each sentence in document and calculates a score for it.
			ArrayList<ArrayList<String>> doc = getSentences(args[1]);

			double total_score = 0.0;

			//Adds to total score for document. 
			for (ArrayList<String> sent : doc) {
				Tree tree = lp.parseStrings(sent); 
				total_score += tree.score(); 
			}
		
			try {	
				//File to keep results in.
				File results_file = new File(args[3]); 
				FileOutputStream fos = new FileOutputStream(results_file); 
				BufferedWriter results_writer = new BufferedWriter(new OutputStreamWriter(fos));			

				results_writer.write("" + total_score);
				results_writer.close();
			}
			catch (IOException e) {
				System.out.println("IO Error!"); 
				System.exit(1); 
			}
		}
	}

	/** Trains a pcfg */
	public static void train(String[] args, boolean self_train) {
		//Sets options for parser. 
		Options op = new Options(); 
		op.doDep = false;
		op.doPCFG = true;
		op.setOptions("-goodPCFG", "-evals", "tsv"); 

		//Sets flags for parser. 
		ArrayList<String> flags = new ArrayList<String>();
		flags.add("-goodPCFG"); 
		flags.add("-evals"); 
		flags.add("tsv");

		//Creates a memory treebank from specified seed set. 
		MemoryTreebank train_treebank = op.tlpParams.memoryTreebank();
		MemoryTreebank in_domain_treebank = op.tlpParams.memoryTreebank();
		MemoryTreebank buffer_treebank = op.tlpParams.memoryTreebank(); 

		String mega_treebank_path = args[1];
			
		//Path for author to put PCFGs in. 
		String author_path = args[2];

		//Author specific treebank. 
		String in_domain_path = author_path + "trees.txt"; 

		//Loads in-domain treebank. Mega treebank loaded below. 
		in_domain_treebank.loadPath(in_domain_path); 

		try { 
			File proportions_file = new File(author_path + "proportions.txt");
			BufferedReader proportion_reader = new BufferedReader(new FileReader(proportions_file));	

			//File to keep all bad trees in. TODO Go through file and add head-rules iff time. 
			File bad_file = new File(author_path + "bad.txt"); 
			FileOutputStream fos = new FileOutputStream(bad_file); 
			BufferedWriter bad_writer = new BufferedWriter(new OutputStreamWriter(fos)); 

			String line = proportion_reader.readLine();

			while (line != null) {
				//Loads treebanks. 
				train_treebank.loadPath(mega_treebank_path);

				String[] proportion_multiplier = line.split(" ");
			
				//Gets the proportion and multiplier 
				String proportion = proportion_multiplier[0].trim();
				int multiplier = Integer.parseInt(proportion_multiplier[1].trim());

				//Bad Indexes to remove from treebank in order to succeed. 
				HashSet<Integer> to_remove = new HashSet<Integer>(); 

				//Looks for bad trees. 
				for (int j = 0; j < in_domain_treebank.size(); j++) {
					try {
						//Attempts to train parser with new tree. 
						buffer_treebank.add(in_domain_treebank.get(j)); 
						LexicalizedParser lp = LexicalizedParser.trainFromTreebank(buffer_treebank, op);

						//Clears buffer lol. 
						buffer_treebank.clear();
					}
					//Queues bad tree for removal. 
					catch (IllegalArgumentException e) {
						to_remove.add(new Integer(j)); 
					}
					catch (IndexOutOfBoundsException e) {
						//This error probably requres manual interference. 
						System.out.println("INDEX OUT OF BOUNDS!"); 
						System.out.println(in_domain_treebank.get(j)); 
						System.exit(1); 
					}
					catch (NullPointerException e) {
						to_remove.add(new Integer(j)); 
					}
				}

				//Adds the good trees!
				for (int i = 0; i < multiplier; i++) {
					for (int j = 0; j < in_domain_treebank.size(); j++) {
						if (!to_remove.contains(new Integer(j)))
							train_treebank.add(in_domain_treebank.get(j)); 
					}
				}

				//Trains from treebank and serializes PCFG. 
				String parser_file_name = proportion.substring(2, 4) + ".pcfg";

				System.out.println("About to train on treebank!!!"); 

				LexicalizedParser lp = LexicalizedParser.trainFromTreebank(train_treebank, op);
				lp.saveParserToSerialized(author_path + parser_file_name);

				//Clears treebank in order to create next one in loop. 
				train_treebank.clear();

				//Gets next line.
				line = proportion_reader.readLine(); 
			}
		}
		catch (IOException e) {
			System.out.println("IO error!"); 
			System.exit(1); 
		}
	}	
	
	public static void parse_corpus(String[] args) {	
		//Sets options for parser. 
		Options op = new Options(); 
		op.doDep = false;
		op.doPCFG = true;
		op.setOptions("-goodPCFG", "-evals", "tsv"); 

		//Sets flags for parser. 
		ArrayList<String> flags = new ArrayList<String>();
		flags.add("-goodPCFG"); 
		flags.add("-evals"); 
		flags.add("tsv");

		//Creates a memory treebank from specified seed set. 
		MemoryTreebank train_treebank = op.tlpParams.memoryTreebank();
		train_treebank.loadPath(args[1]);
		
		//Trains a LexicalizedParser using the treebank. 
		LexicalizedParser lp = LexicalizedParser.trainFromTreebank(train_treebank, op); 

		//Gets list of sentences from self-training set file.  
		String corpus_path = args[2];
		String treebank_path = args[3];

		int self_train_num = 0; //Integer.parseInt(args[4]); 
		ArrayList<ArrayList<String>> sentences = getSentences(corpus_path);  

		//Size argument of 0 specifies to use all sentences.
		if (self_train_num == 0)
			self_train_num = sentences.size(); 

		try{
			//Opens treebank file to write trees to.
			File treebank_file = new File(treebank_path); 
			FileOutputStream fos = new FileOutputStream(treebank_file); 
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fos)); 

			//Parses sentences and adds generated trees to treebank. 
			for (int i = 0; i < self_train_num; i++) {
				ArrayList<String> sentence = sentences.get(i);
				System.out.println(sentence.toString()); 

				Tree tree = lp.parseStrings(sentence);

				writer.write(tree.toString() + "\n"); 
			}

			writer.close(); 
		}
		catch (IOException e) {
				System.out.println("IO Exception error.\n");
				System.exit(1); 
		}
	}

	/** This method returns a list of sentences that is to be read from a 
	  	preprocessed file. This was only used for some of the early data 
		that I preprocessed. */
  	public static ArrayList<ArrayList<String>> getSentences(String filename) {
		BufferedReader reader = null; 
		
		//Opens file. 
		try {
			File read_file = new File(filename);
			reader = new BufferedReader(new FileReader(read_file));
		}
		catch (IOException e) {
			System.out.println("Error opening self-training file!");
			System.exit(1); 
		}

		//Gets list of sentences.
		ArrayList<ArrayList<String>> sentences = new ArrayList<ArrayList<String>>(); 
		ArrayList<String> sentence = new ArrayList<String>(); 

		try {
			String line = reader.readLine();

			while (line != null) {
				sentence = new ArrayList<String>();

				for (String word : line.split("\\s+"))
					sentence.add(word.trim()); 

				sentences.add(sentence); 
				
				line = reader.readLine(); 
			}
		}
		catch (IOException e) {
			System.out.println("Error reading self-training file!"); 
			System.exit(1); 
		}

		return sentences; 
	}
}
