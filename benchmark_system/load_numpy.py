import pickle
f2 = open('output.txt', 'rb')
encoding = pickle.load(f2, encoding='latin1')
f2.close()
f = open('corpus_split.txt')
TEST_SENTENCES = f.readlines()
TEST_SENTENCES = [a.strip() for a in TEST_SENTENCES]

import json
with open('dependency_sentences.txt') as fh:
    data = json.load(fh)

for i in range(len(data)):
    dic = data[i]
    data[i] = {int(k):v for k,v in dic.items()}



set_data = []
for row in data:
    tmp = set()
    for a in row:
        if row[a] in TEST_SENTENCES:
            tmp.add(row[a])
    set_data.append(tmp)


count_len = {}
for a in range(15):
    count_len[a] = 0

for a in vectors:
    count_len[len(a)] += 1



# vectors = []

# for row in data:
#     tmp = []
#     for a in row:
#         if row[a] in TEST_SENTENCES:
#             ind = TEST_SENTENCES.index(row[a])
#             tmp.append(encoding[ind])
#     vectors.append(tmp)


vectors = []

from numpy import array, array_equal, allclose

#test for exact equality
def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if array_equal(elem, myarr)), False)

for row in data:
    tmp = []
    for a in row:
        if row[a] in TEST_SENTENCES:
            ind = TEST_SENTENCES.index(row[a])
            if not arreq_in_list(encoding[ind], tmp):
                tmp.append(encoding[ind])
    vectors.append(tmp)


two_indices = []

for i in range(len(vectors)):
    if (len(vectors[i]) == 2):
        two_indices.append(i)


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





# Experiment settings

DATASET_FP = "./SemEval2018-T4-train-taskA.txt"
# FNAME = './predictions-task' + TASK + '.txt'
# PREDICTIONSFILE = open(FNAME, "w")

K_FOLDS = 10 # 10-fold crossvalidation

# Loading dataset and featurised simple Tfidf-BoW model
corpus, y = parse_dataset(DATASET_FP)


import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F




f2 = open('corpus_dump_output.txt', 'rb')
encoding_vector_append = pickle.load(f2, encoding='latin1')
f2.close()

for i in range(len(vectors)):
    vectors[i].insert(0, encoding_vector_append[i])


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        # layers for query
        self.lin1 = nn.Linear(2304*10, 256)
        self.lin2 = nn.Linear(256, 2)
    def forward(self, x):
        # Query model. The paper uses separate neural nets for queries and documents (see section 5.2).
        # To make it compatible with Conv layer we reshape it to: (batch_size, WORD_DEPTH, query_len)
        x = x.view(-1, 2304*10)
        x1 = self.lin1(x)
        x2 = F.tanh(x1)
        x3 = self.lin2(x2)
        x4 = F.tanh(x3)
        return F.softmax(x4)

model = Module()


import torch.nn.functional as F

import numpy as np
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# criterion = torch.nn.CrossEntropyLoss()

# aa = list(model.parameters())[0].clone()

# concats = np.concatenate((vectors[a][0], vectors[a][1]))
# x = Variable(torch.from_numpy(concats))
# optimizer.zero_grad()
# new_y = np.ndarray(2)
# new_y[0] = 0.00
# new_y[1] = 0.0
# new_y[y[a]] = 1.0
# new_y = Variable(torch.from_numpy(new_y).float())
# n_y = np.ndarray(1)
# n_y[0] = y[a]
# n_y = Variable(torch.from_numpy(n_y).long())
# pred = model(x)
# loss = F.mse_loss(pred.view(1,-1), new_y.view(1,-1))
# # loss = F.nll_loss(pred, n_y)
# loss.backward()
# optimizer.step()

# b = list(model.parameters())[0].clone()

# list(model.parameters())[0].grad

train_till = int(len(vectors)*0.8)
for i in range(1):
    cumulative_loss = 0.0
    # aa = list(model.parameters())[0].clone()
    for a in range(int(train_till)):
        concats = vectors[a][0]
        for vec in vectors[a][1:]:
            concats = np.concatenate((concats, vec))
        to_pad = 2304*10 - concats.shape[0]
        concats = np.concatenate((concats, np.zeros(to_pad)))
        x = Variable(torch.from_numpy(concats).float())
        optimizer.zero_grad()
        new_y = np.ndarray(2)
        new_y[0] = 0.00
        new_y[1] = 0.0
        new_y[y[a]] = 1.0
        new_y = Variable(torch.from_numpy(new_y).float())
        n_y = np.ndarray(1)
        n_y[0] = y[a]
        n_y = Variable(torch.from_numpy(n_y).long())
        pred = model(x)
        # loss = F.mse_loss(pred.view(1,-1), new_y.view(1,-1))
        loss = F.nll_loss(pred, n_y)
        loss.backward()
        optimizer.step()
        cumulative_loss += loss.data[0]
    # b = list(model.parameters())[0].clone()
    # print (torch.equal(aa, b))
    print(cumulative_loss)
    correct_0 = 0
    correct_1 = 0
    notcorrect_0 = 0
    notcorrect_1 = 0
    for a in range(int(train_till), len(vectors), 1):
        concats = vectors[a][0]
        for vec in vectors[a][1:]:
            concats = np.concatenate((concats, vec))
        to_pad = 2304*10 - concats.shape[0]
        concats = np.concatenate((concats, np.zeros(to_pad)))
        x = Variable(torch.from_numpy(concats).float())
        pred = model(x)
        if (y[a] == 0):
            if (pred.data[0][0] > pred.data[0][1]):
                correct_0 += 1
            else:
                notcorrect_0 += 1
        else:
            if (pred.data[0][0] < pred.data[0][1]):
                correct_1 += 1
            else:
                notcorrect_1 += 1
    print("Accuracy", (correct_0 + correct_1) / float(len(vectors) - int(train_till)), (correct_0 + correct_1))



vectors_to_append = []
for a in range(len(vectors), 1):
    concats = vectors[a][0]
    for vec in vectors[a][1:]:
        concats = np.concatenate((concats, vec))
    to_pad = 2304*10 - concats.shape[0]
    concats = np.concatenate((concats, np.zeros(to_pad)))
    x = Variable(torch.from_numpy(concats).float())
    pred = model(x)
    if (y[a] == 0):
        if (pred.data[0][0] > pred.data[0][1]):
            correct_0 += 1
        else:
            notcorrect_0 += 1
    else:
        if (pred.data[0][0] < pred.data[0][1]):
            correct_1 += 1
        else:
            notcorrect_1 += 1
print("Accuracy", (correct_0 + correct_1) / float(len(vectors) - train_till), (correct_0 + correct_1))










# class Module(nn.Module):
#     def __init__(self):
#         super(Module, self).__init__()
#         # layers for query
#         self.lin1 = nn.Linear(2304*9, 2304*4)
#         self.lin11 = nn.Linear(2304*4, 2304)
#         self.lin12 = nn.Linear(2304, 256)
#         self.lin2 = nn.Linear(256, 2)
#     def forward(self, x):
#         # Query model. The paper uses separate neural nets for queries and documents (see section 5.2).
#         # To make it compatible with Conv layer we reshape it to: (batch_size, WORD_DEPTH, query_len)
#         x = x.view(-1, 2304*9)
#         x1 = self.lin1(x)
#         x2 = F.tanh(x1)
#         x1 = self.lin11(x2)
#         x2 = F.tanh(x1)
#         x1 = self.lin12(x2)
#         x2 = F.tanh(x1)
#         x3 = self.lin2(x2)
#         x4 = F.tanh(x3)
#         return F.softmax(x4)

# model = Module()


# import torch.nn.functional as F

# import numpy as np
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# train_till = int(0.8*len(vectors))
# for i in range(8):
#     cumulative_loss = 0.0
#     # aa = list(model.parameters())[0].clone()
#     for a in range(int(train_till)):
#         print(a)
#         concats = vectors[a][0]
#         for vec in vectors[a][1:]:
#             concats = np.concatenate((concats, vec))
#         to_pad = 2304*9 - concats.shape[0]
#         concats = np.concatenate((concats, np.zeros(to_pad)))
#         x = Variable(torch.from_numpy(concats).float())
#         optimizer.zero_grad()
#         new_y = np.ndarray(2)
#         new_y[0] = 0.00
#         new_y[1] = 0.0
#         new_y[y[a]] = 1.0
#         new_y = Variable(torch.from_numpy(new_y).float())
#         n_y = np.ndarray(1)
#         n_y[0] = y[a]
#         n_y = Variable(torch.from_numpy(n_y).long())
#         pred = model(x)
#         # loss = F.mse_loss(pred.view(1,-1), new_y.view(1,-1))
#         loss = F.nll_loss(pred, n_y)
#         loss.backward()
#         optimizer.step()
#         cumulative_loss += loss.data[0]
#     # b = list(model.parameters())[0].clone()
#     # print (torch.equal(aa, b))
#     print(cumulative_loss)
#     correct_0 = 0
#     correct_1 = 0
#     notcorrect_0 = 0
#     notcorrect_1 = 0
#     for a in range(train_till, len(vectors), 1):
#         concats = vectors[a][0]
#         for vec in vectors[a][1:]:
#             concats = np.concatenate((concats, vec))
#         to_pad = 2304*9 - concats.shape[0]
#         concats = np.concatenate((concats, np.zeros(to_pad)))
#         x = Variable(torch.from_numpy(concats).float())
#         pred = model(x)
#         if (y[a] == 0):
#             if (pred.data[0][0] > pred.data[0][1]):
#                 correct_0 += 1
#             else:
#                 notcorrect_0 += 1
#         else:
#             if (pred.data[0][0] < pred.data[0][1]):
#                 correct_1 += 1
#             else:
#                 notcorrect_1 += 1
#     print("Accuracy", (correct_0 + correct_1) / float(len(vectors) - train_till), (correct_0 + correct_1))


import pickle
f2 = open('output_test.txt', 'rb')
encoding = pickle.load(f2, encoding='latin1')
f2.close()
f = open('dependency_dump_new')
TEST_SENTENCES = f.readlines()
TEST_SENTENCES = [a.strip() for a in TEST_SENTENCES]

import json
with open('dependency_test_sentences.txt') as fh:
    data = json.load(fh)

for i in range(len(data)):
    dic = data[i]
    data[i] = {int(k):v for k,v in dic.items()}



set_data = []
for row in data:
    tmp = set()
    for a in row:
        if row[a] in TEST_SENTENCES:
            tmp.add(row[a])
    set_data.append(tmp)


count_len = {}
for a in range(15):
    count_len[a] = 0

for a in vectors:
    count_len[len(a)] += 1



# vectors = []

# for row in data:
#     tmp = []
#     for a in row:
#         if row[a] in TEST_SENTENCES:
#             ind = TEST_SENTENCES.index(row[a])
#             tmp.append(encoding[ind])
#     vectors.append(tmp)


vectors_test = []

from numpy import array, array_equal, allclose

#test for exact equality
def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if array_equal(elem, myarr)), False)

for row in data:
    tmp = []
    for a in row:
        if row[a] in TEST_SENTENCES:
            ind = TEST_SENTENCES.index(row[a])
            if not arreq_in_list(encoding[ind], tmp):
                tmp.append(encoding[ind])
    vectors_test.append(tmp)

f2 = open('test_dump_output.txt', 'rb')
encoding_vector_append = pickle.load(f2, encoding='latin1')
f2.close()

for i in range(len(vectors_test)):
    vectors_test[i].insert(0, encoding_vector_append[i])



out_test = []
for a in range(len(vectors_test)):
    concats = vectors_test[a][0]
    for vec in vectors_test[a][1:]:
        concats = np.concatenate((concats, vec))
    to_pad = 2304*10 - concats.shape[0]
    concats = np.concatenate((concats, np.zeros(to_pad)))
    x = Variable(torch.from_numpy(concats).float())
    pred = model(x)
    out_test.append(pred)

f = open('prediction_test_with_one_more_5.txt', 'w')
for i in out_test:
    if (i.data[0][0] > i.data[0][1]):
        f.write('0\n')
    else:
        f.write('1\n')

f.close()