#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue SEP 25 19:41:24 2016

@author: sherlock
"""


import gensim
import os
import re
# from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
# from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec

def get_doc_list(folder_name):
    doc_list = []
    file_list = [folder_name+'/'+name for name in os.listdir(folder_name) if name.endswith('txt')]
    for file in file_list:
        st = open(file,'r').read()
        doc_list.append(st)
    print ('Found %s documents under the dir %s .....'%(len(file_list),folder_name))
    return doc_list

def get_doc(folder_name):

    doc_list = get_doc_list(folder_name)
    # tokenizer = RegexpTokenizer(r'\w+')
   # en_stop = get_stop_words('en')
    # p_stemmer = PorterStemmer()

    taggeddoc = []

    texts = []
    for index,i in enumerate(doc_list):
        # for tagged doc
        wordslist = []
        tagslist = []

        # clean and tokenize document string
        # raw = i.lower()
        # tokens = tokenizer.tokenize(raw)
        tokens=i.strip().split()
        # remove stop words from tokens
        # stopped_tokens = [i for i in tokens if not i in en_stop]

        # remove numbers
        number_tokens = [re.sub(r'[\d]', ' ', i) for i in tokens]
        number_tokens = ' '.join(number_tokens).split()

        # stem tokens
        # stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
        # remove empty
        
        td = TaggedDocument((' '.join(number_tokens)).split(),str(index))


        taggeddoc.append(td)

    print ('Get all data!')
    return taggeddoc


#load foldname
afterCutFilePath='/yourfilepath'
documents = get_doc(afterCutFilePath)
print ('Data Loading finished')
print (len(documents),type(documents))

# build the model
print ('start training!')
model = Doc2Vec(documents, dm = 1, negative =5 , size= 100,window = 10, min_alpha=0.025, min_count=1,workers=4)
model.save('./train_seg2.model')
#from gensim.models.word2vec import Word2Vec, Sent2Vec, LineSentence
#model=Sent2Vec(LineSentence(sent_file),size=100, window=5, sg=0, min_count=5, workers=8)
#model.save('./train_sent2vec.vec')
