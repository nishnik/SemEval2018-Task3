import argparse
import codecs
import os
import re
import sys
import io


filename = "test-dump.txt.predict"

f = codecs.open(filename, "r", "utf-8")
corpus = []
sentence = []

for line in f:
    if line.strip() == "":
        corpus.append(sentence)
        sentence = []
        continue
    else:
        line = line.strip()
        cline = line.split(u"\t")
        sentence.append(cline)

f.close()


# for a in corpus:
#     for b in a:
#         if (b[2] != '_'):
#             print (b)

# b[2] is not significant

# for a in corpus:
#     for b in a:
#         del b[2]

# for a in corpus:
#     for b in a:
#         if (b[2] != b[3]):
#             print (b)

# Nothing prints here too

# for a in corpus:
#     for b in a:
#         del b[2]


dep_corpus = []
for a in corpus:
    sentence = []
    for b in a:
        tmp = []
        tmp.append(int(b[0]))
        tmp.append(b[1])
        tmp.append(int(b[6]))
        sentence.append(tmp)
    dep_corpus.append(sentence)

id_to_word_corpus = []

for a in dep_corpus:
    id_to_word = {}
    for b in a:
        id_to_word[b[0]] = b[1]
    id_to_word_corpus.append(id_to_word)

graphs = []

for a in dep_corpus:
    graph = {}
    for b in a:
        graph[b[0]] = b[2]
    graphs.append(graph)


dependencies = []

for graph in graphs:
    roots = []
    roots.append(-1)
    for edge in graph:
        if (graph[edge] == 0):
            roots.append(edge)
    roots_dict = {}
    for a in roots:
        if a == -1:
            roots_dict[a] = []
        else:    
            roots_dict[a] = [a]
    change = True
    while (change != False):
        change = False
        for edge in graph:
            if (graph[edge] in roots and not edge in roots_dict[graph[edge]]):
                change = True
                roots_dict[graph[edge]].append(edge)
                for edge_again in graph:
                    if (graph[edge_again] == edge):
                        graph[edge_again] = graph[edge]
    dependencies.append(roots_dict)

for a in dependencies:
    for b in a:
        a[b].sort()

dependencies_sentences = []

for i in range(len(dependencies)):
    sentences = {}
    for root in dependencies[i]:
        listed = dependencies[i][root]
        sentence = ' '.join([id_to_word_corpus[i][x] for x in listed])
        sentences[root] = sentence
    dependencies_sentences.append(sentences)

import json
with open('dependency_test_sentences.txt', 'w') as outfile:
     json.dump(dependencies_sentences, outfile, sort_keys = True, indent = 4,
               ensure_ascii = False)
