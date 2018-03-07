
# ----------------------------------------------------------
# preprocessSentences+.py
# code used from the following sources:
# https://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/
# https://stevenloria.com/tf-idf/
# https://docs.scipy.org/doc/numpy/reference (for dealing with sparse matrices)
#
# Program is run using the following command. The txt file with review documents 
# as well as the featur extraction variation number is hard coded
# Along with previously specified files, the program also returns an npz file with 
# a sparse matrix of TF-IDF scores.
# python ./preprocessSentences+.py -p . -o var7 
# python ./preprocessSentences+.py -p . -o test -v var1_vocab_5.txt
# ----------------------------------------------------------

# In[ ]:

# loading the npz file to run classifiers:
# data = np.load('123.npz')


import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import numpy
import re
import sys
import getopt
import codecs
import time
import os
import csv

# -- bigram imports --
import itertools
from nltk.metrics import BigramAssocMeasures

# spellcheck library
from autocorrect import spell

# -- TF-IDF imports --
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse

# -- bigram imports --
import nltk
from nltk.collocations import *


# In[ ]:


chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']

chars2 = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '.', ';', 
'*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']


# In[ ]:


def stem(word):
   regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
   stem, suffix = re.findall(regexp, word)[0]
   return stem


# In[ ]:


def unique(a):
   """ return the list with duplicate elements removed """
   return list(set(a))


# In[ ]:


def intersect(a, b):
   """ return the intersection of two lists """
   return list(set(a) & set(b))
def union(a, b):
   """ return the union of two lists """
   return list(set(a) | set(b))


# In[ ]:


def get_files(mypath):
   return [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

def get_dirs(mypath):
   return [ f for f in listdir(mypath) if isdir(join(mypath,f)) ]


# In[ ]:


# Reading a bag of words file back into python. The number and order
# of sentences should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile):
  bagofwords = numpy.genfromtxt('myfile.csv',delimiter=',')
  return bagofwords


# In[ ]:



def tokenize_corpus(path, train=True):

  # hardcode the bag of words variation 
  var = 7;
  # var1 - simple
  # var2 - lowercase
  # var3 - lowercase + no punctuation, except '!' and '?'
  # var4 - lowercase + no punctuation
  # var5 - lowercase + no punctuation + stopwords
  # var6 - lowercase + no punctuation + stopwords + stems/lemmas
  # var7 - lowercase + no punctuation + stopwords + stems/lemmas + spellcheck

  # used to stem words later on/get roots of words
  porter = nltk.PorterStemmer() # also lancaster stemmer
  # used to get lemmas (complete words themselves) of words
  wnl = nltk.WordNetLemmatizer()
  # list of stopwords, can print to view
  stopWords = stopwords.words("english")
  classes = []
  samples = []
  docs = []
  if train == True:
    words = {}
  f = open(path, 'r')
  lines = f.readlines()

  for line in lines:
    # separating serial, review and label
    classes.append(line.rsplit()[-1])
    samples.append(line.rsplit()[0])
    raw = line.decode('latin1')
    raw = ' '.join(raw.rsplit()[1:-1])

    if (var == 3):
    # remove noise characters, except for '!' and '?'
      raw = re.sub('[%s]' % ''.join(chars2), ' ', raw)

    if (var > 3):
    # remove noisy characters; tokenize - specified at the start
      raw = re.sub('[%s]' % ''.join(chars), ' ', raw)

    tokens = word_tokenize(raw)
    
    if (var > 1):
    # make lower case - !! consider all capitals as more positive or negative?
    # !! what if we didn't do this?
      tokens = [w.lower() for w in tokens]

    if (var > 6):
    # perform spell check - before stems,lemmas and stopwords
      tokens = [spell(w) for w in tokens]

    if (var > 4):
    # removing stopwords
    # using python stopwords scripts - !! add manually to stopwords?
    # !! what if we didn't do this?
      tokens = [w for w in tokens if w not in stopWords]

    if (var > 5):
    # first lemmatize then stem, significance?
    # !! what if we didn't do this? - lemmatize
      tokens = [wnl.lemmatize(t) for t in tokens]
    # !! what if we didn't do this? - stem    
      tokens = [porter.stem(t) for t in tokens] 


    # ---------------------------------------------------------
    # EDIT THIS BIT TO USE BIGRAMS
    # !! add bigram collocations here
    # tokens = bigram_tokenize_corpus(tokens, BigramAssocMeasures.chi_sq, 200)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    # !! experiment with window size for accuracy
    finder = BigramCollocationFinder.from_words(tokens, window_size = 4)
    # !! experiment with frequency count for accuracy 
    finder.apply_freq_filter(2)
    # !! understand pmi filter 
    # bigrams = finder.nbest(bigram_measures.pmi, 10)
    
    #for k,v in finder.ngram_fd.items():
        #print(k,v)

    #for k in finder.ngram_fd.items():
      #print(k)

    # ---------------------------------------------------------

    if train == True:
        # add to word count frequency
     for t in tokens: 
         try:
             words[t] = words[t]+1
         except:
             words[t] = 1
    docs.append(tokens)

  # docs: consist of a long list of tokens as they
  # appear in the text ie may be repeated
  # words: dictionary - has the frequency of the vocabulary

  if train == True:
     return(docs, classes, samples, words)
  else:
     return(docs, classes, samples)

# -----------------------------------------------------------------
# In[ ]:
# this is the bigram method - it needs major edits
 
def bigram_tokenize_corpus(tokens, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    # !! selecting within the bigrams
    # bigrams = bigram_finder.nbest(score_fn, n)
    tokens = [ngram for ngram in itertools.chain(tokens, bigrams)]
    
    # return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
    return(tokens)
 
# evaluate_classifier(bigram_word_feats)
# -----------------------------------------------------------------

# In[ ]:


# ?? !! why is the wordcount threshold hardcoded?
def wordcount_filter(words, num=5):
   keepset = []
   for k in words.keys():
       if(words[k] > num):
           keepset.append(k)
   print "Vocab length:", len(keepset)
   return(sorted(set(keepset)))


# In[ ]:


def find_wordcounts(docs, vocab):
   bagofwords = numpy.zeros(shape=(len(docs),len(vocab)), dtype=numpy.uint8)
   vocabIndex={}
   # giving each vocab word an index value in a dictionary
   for i in range(len(vocab)):
      vocabIndex[vocab[i]]=i

   for i in range(len(docs)):
       doc = docs[i]

       for t in doc:
          index_t=vocabIndex.get(t)
          # if the vocab word is present in the known words
          if index_t>=0:
            # i is the doc number
            # for a particular doc, increment count in bow matrix
             bagofwords[i,index_t]=bagofwords[i,index_t]+1

   print "Finished find_wordcounts for:", len(docs), "docs"
   return(bagofwords)

# -----------------------------------------------------------------
# In[ ]:
# Working on TF - IDF


def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    return 1 + math.log(tokenized_document.count(term))

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))

def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def find_tfidf(docs, vocab):
   idf = inverse_document_frequencies(docs)
   bagofwords = numpy.zeros(shape=(len(docs),len(vocab)), dtype=numpy.uint8)
   vocabIndex={}
   # giving each vocab word an index value in a dictionary
   for i in range(len(vocab)):
      vocabIndex[vocab[i]]=i

   for i in range(len(docs)):
       doc = docs[i]
       for t in doc:
          tf = sublinear_term_frequency(t, doc)
          index_t=vocabIndex.get(t)
          # if the vocab word is present in the known words
          if index_t>=0:
            # i is the doc number
            # for a particular doc, enter tfidf score
             bagofwords[i,index_t]= (tf * idf[t])

   print "Finished find_tfidf for:", len(docs), "docs"
   # print bagofwords
   # return(bagofwords)

   idf = inverse_document_frequencies(docs)
   tfidf_documents = []
   for document in docs:
      doc_tfidf = []
      for term in idf.keys():
         tf = augmented_term_frequency(term, document)
         doc_tfidf.append(tf * idf[term])
      tfidf_documents.append(doc_tfidf)
   
   return tfidf_documents

# -----------------------------------------------------------------

# In[ ]:


def main(argv):
  
  start_time = time.time()

  path = ''
  outputf = 'out'
  vocabf = ''

  try:
   opts, args = getopt.getopt(argv,"p:o:v:",["path=","ofile=","vocabfile="])
  except getopt.GetoptError:
    print 'Usage: \n python preprocessSentences.py -p <path> -o <outputfile> -v <vocabulary>'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print 'Usage: \n python preprocessSentences.py -p <path> -o <outputfile> -v <vocabulary>'
      sys.exit()
    elif opt in ("-p", "--path"):
      path = arg
    elif opt in ("-o", "--ofile"):
      outputf = arg
    elif opt in ("-v", "--vocabfile"):
      vocabf = arg

  # !! edit for train data!!!
  traintxt = path+"/train.txt"
  print 'Path:', path
  print 'Training data:', traintxt

  # Tokenize training data (if training vocab doesn't already exist):
  if (not vocabf):
    # !! change threshold?
    word_count_threshold = 5
    # classes = labels, samples = sample number, 
    (docs, classes, samples, words) = tokenize_corpus(traintxt, train=True)
    vocab = wordcount_filter(words, num=word_count_threshold)
    # Write new vocab file
    vocabf = outputf+"_vocab_"+str(word_count_threshold)+".txt"
    outfile = codecs.open(path+"/"+vocabf, 'w',"utf-8-sig")
    outfile.write("\n".join(vocab))
    outfile.close()
  else:
    word_count_threshold = 0
    (docs, classes, samples) = tokenize_corpus(traintxt, train=False)
    vocabfile = open(path+"/"+vocabf, 'r')
    vocab = [line.rstrip('\n') for line in vocabfile]
    vocabfile.close()

  print 'Vocabulary file:', path+"/"+vocabf 

  # Get bag of words:
  bow = find_wordcounts(docs, vocab)
  # Check: sum over docs to check if any zero word counts
  print "Doc with smallest number of words in vocab has:", min(numpy.sum(bow, axis=1))

  # -----------------------------------------------------------
  # Get tfidf
  # tfidf_representation = find_tfidf(docs, vocab)

  # Use SkiKit learn for tf-idf
  # sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True)

  # sklearn_representation = sklearn_tfidf.fit_transform(docs)
  # print sklearn_representation.toarray()[0].tolist()
  transformer = TfidfTransformer(smooth_idf=False)
  tfidf = transformer.fit_transform(bow)
  tfidf.todense()
  scipy.sparse.save_npz(outputf+"_tfidf_"+str(word_count_threshold)+".npz", tfidf)
  # print(numpy.matrix(tfidf))
  # print tfidf
  

  # Write bow file
  with open(path+"/"+outputf+"_bag_of_words_"+str(word_count_threshold)+".csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(bow)

  # Write tfidf file - !!! problem: currently doesn't record all values 
  #with open(path+"/"+outputf+"_tfidf_"+str(word_count_threshold)+".csv", "wb") as f:
    #writer = csv.writer(f)
    #writer.writerows(tfidf)

  # Write classes
  outfile = open(path+"/"+outputf+"_classes_"+str(word_count_threshold)+".txt", 'w')
  outfile.write("\n".join(classes))
  outfile.close()

  # Write samples
  outfile = open(path+"/"+outputf+"_samples_class_"+str(word_count_threshold)+".txt", 'w')
  outfile.write("\n".join(samples))
  outfile.close()

  print 'Output files:', path+"/"+outputf+"*"

  # Runtime
  print 'Runtime:', str(time.time() - start_time)

if __name__ == "__main__":
  main(sys.argv[1:])

