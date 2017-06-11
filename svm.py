from dataset_metallic_glass import PartDataset, PartDatasetSVM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


d = PartDatasetSVM(root = 'mg', classification = False)
dt = PartDatasetSVM(root = 'mg', classification = False, train = False)


l = len(d)
print(len(d.classes), l)


lt = len(dt)
print(lt)

train_set = []
train_label = []
for i in range(l):
    idx = i
    sample, label = d[idx]
    train_set.append(sample)
    train_label.append(label)
    if i%100 == 0:
        print i
    
test_set = []
test_label = [] 
for i in range(lt):
    idx = i
    sample, label = dt[idx]
    test_set.append(sample)
    test_label.append(label)
    if i%100 == 0:
        print i
        
        
train_set = np.array(train_set)
train_label = np.array(train_label)
test_set = np.array(test_set)
test_label = np.array(test_label)

print(train_set.shape, train_label.shape)

from sklearn.svm import SVC
clf = SVC()

clf.fit(train_set, train_label)

accuracy = np.sum(clf.predict(train_set) == train_label)/float(l)
print(accuracy)

accuracy_test = np.sum(clf.predict(test_set) == test_label)/float(lt)
print(accuracy_test)