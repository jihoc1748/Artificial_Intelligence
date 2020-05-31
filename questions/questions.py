import nltk
import sys
import os
#for string.punctuation which is list of all punctuations
import string
import math


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    while True:
        query = set(tokenize(input("Query: ")))

        # Determine top file matches according to TF-IDF
        filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

        # Extract sentences from top files
        sentences = dict()
        for filename in filenames:
            for passage in files[filename].split("\n"):
                for sentence in nltk.sent_tokenize(passage):
                    tokens = tokenize(sentence)
                    if tokens:
                        sentences[sentence] = tokens

        # Compute IDF values across sentences
        idfs = compute_idfs(sentences)

        # Determine top sentence matches
        matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
        for match in matches:
            print(match)
        print("\n"*3)
        a = input("continue y/n: ")
        a = a.lower()
        if "n" in a:
            break


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    data = {}

    #loop over every subdirectry in directory i.e i= abc.txt
    for i in os.listdir(directory):
        #path independent join
        path = os.path.join(directory,f"{(i)}")
        #open individual file
        file = open(path)
        #read through entire passage and remove line break
        line = file.read().replace("\n", " ")
        #closing file
        file.close()
        #add it to data dictionary
        data[i] = line

    return data


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    #splitting list smartly using nltk word_tokenize
    words = nltk.word_tokenize(document)

    #index of words to be filtered out
    rem = []
    #filtering part
    for i in range(len(words)):
        #lowercase conversion
        words[i] = words[i].lower()

        #stopwords like in out aint ... eg. , 'few', 'more', 'most', 'other', 'so
        #string.punctuation gives list '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        if words[i] in nltk.corpus.stopwords.words("english") or words[i] in string.punctuation:
            rem.append(words[i])
            continue

    #removing filtered filtered words
    for i in rem:
        words.remove(i)


    return words



def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = {}

    #looping through every directory
    for i in documents:
        #content = list of words in that key of document where key is name of directory
        content = documents[i]
        #looping over every word in content

        for word in content:
            #if word already in words then going to next word
            if word in words:
                continue
            else:
                #count  = number of document having word
                count = 0
                #total = number of documents
                total = 0
                #looping entire directories again
                for temp in documents:
                    #checking word in the document
                    if word in documents[temp]:
                        count +=1
                    total +=1
                #adding  inverse document frequency value
                #math.log is log with base e or natural log
                words[word] = math.log(float(total/count))

    return words

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = {}

    #looping over documents:
    for file in files:

        #sum  tf-idf value total count of word-inverse document frequency value
        sum = 0
        #looping over words in Query
        for word in query:

                #inverse document frequency value of word
                idf = idfs[word]

                #tf*idf
                sum += files[file].count(word)*idf
        tf_idf[file] = sum

    #ranked files accordng to tf-idf values
    rank = sorted(tf_idf.keys(), key=lambda x: tf_idf[x], reverse = True)

    rank = list(rank)
    try:
        return rank[0:n]
    except:
        return rank

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idf = {}
    #looping over every sentence
    for sent in sentences:

        #sum of idfs values
        sum = 0
        #list of words in sentence
        words = sentences[sent]

        #number of words in sentence
        count = len(words)
        #number of times the word occur in sentence
        word_count = 0
        #looping of every word in query
        for word in query:
            #if word in query also in sentence add idf to sum
            if word in words:
                sum += idfs[word]
                word_count += 1

        idf[sent] = (sum,float(word_count/count),)

    rank = sorted(idf.keys(), key=lambda x: idf[x], reverse = True)
    rank = list(rank)
    try:
        return rank[0:n]
    except:
        return rank
if __name__ == "__main__":
    main()
