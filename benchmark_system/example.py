#!/usr/bin/env python3
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import dump_svmlight_file
from sklearn import metrics
import numpy as np
import logging
import codecs

logging.basicConfig(level=logging.INFO)

def parse_dataset(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with open(fp, 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)
    return corpus, y



def parse_test(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with open(fp, 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                # label = int(line.split("\t")[1])
                tweet = line.split("\t")[1]
                # y.append(label)
                corpus.append(tweet)
    return corpus



tset_new = "./test_data"
# Experiment settings

DATASET_FP = "./SemEval2018-T4-train-taskA.txt"
# FNAME = './predictions-task' + TASK + '.txt'
# PREDICTIONSFILE = open(FNAME, "w")

K_FOLDS = 10 # 10-fold crossvalidation

# Loading dataset and featurised simple Tfidf-BoW model
corpus, y = parse_dataset(DATASET_FP)

tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True).tokenize
vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
X = vectorizer.fit_transform(corpus)
# vectorizer.get_feature_names()
class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()

# Returns an array of the same size as 'y' where each entry is a prediction obtained by cross validated
CLF = LinearSVC() # the default, non-parameter optimized linear-kernel SVM


predicted = cross_val_predict(CLF, X, y, cv=K_FOLDS)

# Modify F1-score calculation depending on the task

print ("F1-score Task",  metrics.f1_score(y, predicted, pos_label=1))
print ("F1-score Task",  metrics.precision_score(y, predicted, pos_label=1))
print ("F1-score Task",  metrics.recall_score(y, predicted, pos_label=1))

# for p in predicted:
#     PREDICTIONSFILE.write("{}\n".format(p))
# PREDICTIONSFILE.close()
    
    
    
