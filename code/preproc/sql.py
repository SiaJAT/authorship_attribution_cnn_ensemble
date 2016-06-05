#!/usr/bin/python

import MySQLdb
import sys
import numpy as np

# Usage to create database: python sql.py create_db 
# Usage to create one table: python sql.py build_table <"GLOVE" or "WORD2VEC"> <path to vectors>

class NLP_Database:
    def __init__(self):
        reload(sys)
        sys.setdefaultencoding('utf8')
        
        #Login info. 
        self.host = "localhost"
        self.user = "root"
        self.passwd = "julian"

        #Keeps track of table selection. 
        self.table = None
        self.num_rows = None
        self.rows_fetched = 0

        #Buffers previously fetched word for searches. 
        self.row_buffer = None

        #Variables for table building. 
        self.max_word_len = 250

        #Opens database connection. 
        self.db = MySQLdb.connect(self.host, self.user, self.passwd)

        #Cursor object used to interact with MySQL
        self.cursor = self.db.cursor()

        #Selects db. 
        self.cursor.execute("USE NLP_VECTORS")

    def __del__(self):
        #Disconnects from database. 
        self.db.close()

    #This method loads one of the tables and all of its entries. 
    #Used before fetching entries. 
    def pick_table(self, table):
        if not (table == "GLOVE" or table == "WORD2VEC"):
            print "Incorrect table requested, try GLOVE or WORD2VEC"
            System.exit()
    
        #Pre-loads all entries and readies for fetching. 
        sql = "SELECT * FROM %s" % (table)
        self.num_rows = self.cursor.execute(sql)

        #Denotes that table has been selected. 
        self.table = table

    #Fetches next row from currently selected table.
    #Returns None if no next row available. 
    def fetch_one(self):
        if self.table == None:
            print "No table selected! Did you forget to call pick_table()?" 
            System.exit()
       
        #Returns None if no more rows to read. 
        if self.rows_fetched > self.num_rows:
            return None

        #Increments num fetched and returns row. 
        self.rows_fetched += 1
        
        return self.cursor.fetchone()
        
    def get_wordvec(self, word):
        done = False
        word_list = None

        #Checks if word we're looking for was buffered already. 
        if self.row_buffer is not None and self.row_buffer[0] == word:
            word_list = self.row_buffer

            done = True

        while not done:
            row = self.fetch_one()
            
            #Word not in database. 
            if row == None or row[0] > word: 
                #Buffer for next query. 
                self.row_buffer = row

                #Safety
                done = True

                #Create random gaussian vector.
                # TODO Change 300 if vector dimmensionality changes. 
                return np.random.normal(0, 0.1, 300)
            
            if row[0] == word:
                done = True

                #Creates numpy vector for word_list and returns it. 
                vector = np.array([float(entry) for entry in row[1].split()])

                #Case where vector is malformed. 
                if len(vector) != 300:
                    print "Vector dimmensions for " + word + " in db bad!"
                    print word + " vector: " + row[1]

                    #Returns random gaussian assuming # of errors is small. 
                    return np.random.normal(0, 0.1, 300)

                return vector
                

#Deletes database for rebuilding. 
def clean():
    sql = "DROP DATABASE IF EXISTS NLP_VECTORS"

    cursor.execute(sql)
    
    #Commits changes. 
    db.commit()

#Creates database. 
def create_db():
    #Cleans before building. 
    clean()

    #Creates database if it doesn't exist. 
    sql = "CREATE DATABASE IF NOT EXISTS NLP_VECTORS"

    cursor.execute(sql)

    #Selects database.
    sql = "USE NLP_VECTORS"

    cursor.execute(sql)

    #Creates vector tables if doesn't exist yet. 
    sql = """CREATE TABLE IF NOT EXISTS WORD2VEC (
             WORD VARCHAR(%d) NOT NULL,
             VECTOR MEDIUMTEXT NOT NULL )""" % (max_word_len)

    cursor.execute(sql)

    sql = """CREATE TABLE IF NOT EXISTS GLOVE (
             WORD VARCHAR(%d) NOT NULL, 
             VECTOR MEDIUMTEXT NOT NULL )""" % (max_word_len)

    cursor.execute(sql)

    #Commits changes. 
    db.commit()

#Builds either the GLOV or WORD2VEC tables in the db. 
def build_table(table_name, file_name):
    if not (table_name == "GLOVE" or table_name == "WORD2VEC"):
        print "Wrong table name used! Try GLOVE or WORD2VEC silly."
  
        sys.exit()

    #Selects database for use. 
    sql = "USE NLP_VECTORS"
    cursor.execute(sql)

    #Adds each vector to table. 
    word_file = open(file_name, 'r')

    for line in word_file:
        word, vector = line.split(' ', 1)

        #Escapes apostrophe and backslash by doubling them. 
        word = word.replace("'", "''").replace("\\", "\\\\")

        sql = """INSERT INTO %s (WORD, VECTOR) 
                 VALUES('%s', '%s')""" % (table_name, word, vector)

        cursor.execute(sql)

    #Orders table in alphabetical order by word. 
    sql = """ALTER TABLE %s ORDER BY WORD""" % (table_name)
    cursor.execute(sql)

    #Commits changes. 
    db.commit()

    word_file.close()

if __name__ == "__main__":
    #Ensures argument length correctness. 
    if len(sys.argv) < 2:
        print "Need at least one argument!"

    #Command to execute in db. 
    command = sys.argv[1]

    #Different command cases. 
    if command == "clean":
        clean()
    elif command == "create_db":
        create_db()
    elif command == "build_table":
        #Ensures arguments are all there. 
        if not len(sys.argv) == 4:
            print "Expecting 4 arguments!"
            print "Usage: $python sql.py build_table [table name] [word vector file name]"
        else:
            build_table(sys.argv[2], sys.argv[3])

    db.close()
