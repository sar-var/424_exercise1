# -------------
# prepreprocess.py
# run using:
# -------------

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
import nltk
from nltk.collocations import *

# In[ ]:


chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']


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
    # remove noisy characters; tokenize - specified at the start
    raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
    tokens = word_tokenize(raw)
    # make lower case - !! consider all capitals as more positive or negative?
    # !! what if we didn't do this?
    tokens = [w.lower() for w in tokens]
    # removing stopwords
    # using python stopwords scripts - !! add manually to stopwords?
    # !! what if we didn't do this?
    tokens = [w for w in tokens if w not in stopWords]
    # first lemmatize then stem, significance?
    # !! what if we didn't do this? - lemmatize
    tokens = [wnl.lemmatize(t) for t in tokens]
    # !! what if we didn't do this? - stem
    tokens = [porter.stem(t) for t in tokens] 

    # ---------------------------------------------------------
    # !! add bigram collocations here
    # tokens = bigram_tokenize_corpus(tokens, BigramAssocMeasures.chi_sq, 200)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    # !! experiment with window size for accuracy
    finder = BigramCollocationFinder.from_words(tokens, window_size = 3)
    # !! experiment with frequency count for accuracy 
    bigrams = finder.apply_freq_filter(2)
    # !! understand pmi filter 
    bigrams = finder.nbest(bigram_measures.pmi, 10)
    
    for k,v in finder.ngram_fd.items():
        print(k,v)

    # append is not working, redo
    #print bigrams
    # ---------------------------------------------------------

    if train == True:
        # add to word count frequency
      for t in tokens: 
        # convert bigram tuple to string
        if isinstance(t, (list,)):
            t = ' '.join(str(d) for d in t)
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
   for i in range(len(vocab)):
      vocabIndex[vocab[i]]=i

   for i in range(len(docs)):
       doc = docs[i]

       for t in doc:
          index_t=vocabIndex.get(t)
          if index_t>=0:
             bagofwords[i,index_t]=bagofwords[i,index_t]+1

   print "Finished find_wordcounts for:", len(docs), "docs"
   return(bagofwords)


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

  traintxt = path+"/train3000.txt"
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
'''
  # Get bag of words:
  bow = find_wordcounts(docs, vocab)
  # Check: sum over docs to check if any zero word counts
  print "Doc with smallest number of words in vocab has:", min(numpy.sum(bow, axis=1))

  # Write bow file
  with open(path+"/"+outputf+"_bag_of_words_"+str(word_count_threshold)+".csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(bow)

  # Write classes
  outfile = open(path+"/"+outputf+"_classes_"+str(word_count_threshold)+".txt", 'w')
  outfile.write("\n".join(classes))
  outfile.close()

  # Write samples
  outfile = open(path+"/"+outputf+"_samples_class_"+str(word_count_threshold)+".txt", 'w')
  outfile.write("\n".join(samples))
  outfile.close()

  print 'Output files:', path+"/"+outputf+"*"
 '''

  # Runtime
  # print 'Runtime:', str(time.time() - start_time)

if __name__ == "__main__":
  main(sys.argv[1:])

