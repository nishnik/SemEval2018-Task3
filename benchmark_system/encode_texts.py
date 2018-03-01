# -*- coding: utf-8 -*-

""" Use DeepMoji to encode texts into emotional feature vectors.
"""
from __future__ import print_function, division
import example_helper
import json
import csv
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_feature_encoding
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

f = open('deepemoji/DeepMoji/dependency_dump_new')
TEST_SENTENCES = f.readlines()
TEST_SENTENCES = [unicode(a.strip(), 'utf-8') for a in TEST_SENTENCES]

# TEST_SENTENCES = [u'I love mom\'s cooking',
#                   u'I love how you never reply back..',
#                   u'I love cruising with my homies',
#                   u'I love messing with yo mind!!',
#                   u'I love you and now you\'re just gone..',
#                   u'This is shit',
#                   u'This is the shit']

maxlen = 30
batch_size = 32

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary, maxlen)
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = deepmoji_feature_encoding(maxlen, PRETRAINED_PATH)
model.summary()

print('Encoding texts..')
encoding = model.predict(tokenized)

print('First 5 dimensions for sentence: {}'.format(TEST_SENTENCES[0]))
print(encoding[0,:5])

# Now you could visualize the encodings to see differences,
# run a logistic regression classifier on top,
# or basically anything you'd like to do.



import pickle
f=open('output_test.txt','w')
pickle.dump(encoding, f)
f.close()



f2 = open('output.txt', 'rb')
encoding = pickle.load(f2, encoding='latin1')
f2.close()