#!/usr/bin/env python3
from nltk.tokenize import TweetTokenizer
import numpy as np
import logging
import codecs

logging.basicConfig(level=logging.INFO)

def parse_dataset_test(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    corpus = []
    with open(fp, 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                corpus.append(tweet)
    return corpus






DATASET_FP = "./test_data"

corpus = parse_dataset_test(DATASET_FP)
#corpus = corpus[1:]
f =  open('test-dump.txt', 'w')
for a in corpus:
    f.write(a + "\n")

f.close()