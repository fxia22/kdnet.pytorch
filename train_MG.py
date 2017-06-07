from dataset_metallic_glass import PartDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from kdnet import KDNet_Batch_mp as KDNet_Batch

def split_ps(point_set):
    #print point_set.size()
    num_points = point_set.size()[0]/2
    diff = point_set.max(dim=0)[0] - point_set.min(dim=0)[0]
    dim = torch.max(diff, dim = 1)[1][0,0]
    cut = torch.median(point_set[:,dim])[0][0]
    left_idx = torch.squeeze(torch.nonzero(point_set[:,dim] > cut))
    right_idx = torch.squeeze(torch.nonzero(point_set[:,dim] < cut))
    middle_idx = torch.squeeze(torch.nonzero(point_set[:,dim] == cut))

    if torch.numel(left_idx) < num_points:
        left_idx = torch.cat([left_idx, middle_idx[0:1].repeat(num_points - torch.numel(left_idx))], 0)
    if torch.numel(right_idx) < num_points:
        right_idx = torch.cat([right_idx, middle_idx[0:1].repeat(num_points - torch.numel(right_idx))], 0)

    left_ps = torch.index_select(point_set, dim = 0, index = left_idx)
    right_ps = torch.index_select(point_set, dim = 0, index = right_idx)
    return left_ps, right_ps, dim


def split_ps_reuse(point_set, level, pos, tree, cutdim):
    sz = point_set.size()
    num_points = np.array(sz)[0]/2
    max_value = point_set.max(dim=0)[0]
    min_value = -(-point_set).max(dim=0)[0]

    diff = max_value - min_value
    dim = torch.max(diff, dim = 1)[1][0,0]

    cut = torch.median(point_set[:,dim])[0][0]
    left_idx = torch.squeeze(torch.nonzero(point_set[:,dim] > cut))
    right_idx = torch.squeeze(torch.nonzero(point_set[:,dim] < cut))
    middle_idx = torch.squeeze(torch.nonzero(point_set[:,dim] == cut))

    if torch.numel(left_idx) < num_points:
        left_idx = torch.cat([left_idx, middle_idx[0:1].repeat(num_points - torch.numel(left_idx))], 0)
    if torch.numel(right_idx) < num_points:
        right_idx = torch.cat([right_idx, middle_idx[0:1].repeat(num_points - torch.numel(right_idx))], 0)

    left_ps = torch.index_select(point_set, dim = 0, index = left_idx)
    right_ps = torch.index_select(point_set, dim = 0, index = right_idx)

    tree[level+1][pos * 2] = left_ps
    tree[level+1][pos * 2 + 1] = right_ps
    cutdim[level][pos * 2] = dim
    cutdim[level][pos * 2 + 1] = dim

    return

test = False
import sys
if len(sys.argv) > 1 and sys.argv[1] == 'test':
    test = True
    d = PartDataset(root = 'mg', classification = True, train = False)
else:
    d = PartDataset(root = 'mg', classification = True)


l = len(d)
print(len(d.classes), l)
levels = (np.log(2048)/np.log(2)).astype(int)
net = KDNet_Batch().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

if test:
    net.load_state_dict(torch.load(sys.argv[2]))
    net.eval()

sum_correct = 0
sum_sample = 0


for it in range(10000):
    optimizer.zero_grad()
    losses = []
    corrects = []
    points_batch = []
    cutdim_batch = []
    targets = []
    bt = 20
    for batch in range(bt):
        j = np.random.randint(l)
        point_set, class_label = d[j]
        #print(point_set, class_label)
        targets.append(class_label)

        if batch == 0 and it ==0:
            tree = [[] for i in range(levels + 1)]
            cutdim = [[] for i in range(levels)]
            tree[0].append(point_set)

            for level in range(levels):
                for item in tree[level]:
                    left_ps, right_ps, dim = split_ps(item)
                    tree[level+1].append(left_ps)
                    tree[level+1].append(right_ps)
                    cutdim[level].append(dim)
                    cutdim[level].append(dim)

        else:
            tree[0] = [point_set]
            for level in range(levels):
                for pos, item in enumerate(tree[level]):
                    split_ps_reuse(item, level, pos, tree, cutdim)
                        #print level, pos

        #cutdim_v = [(torch.from_numpy(np.array(item).astype(np.int64))) for item in cutdim]
        cutdim_v = [(torch.from_numpy(np.array(item).astype(np.int64))) for item in cutdim]
        points = torch.stack(tree[-1])
        points_batch.append(torch.unsqueeze(torch.squeeze(points), 0).transpose(2,1))
        cutdim_batch.append(cutdim_v)

    points_v = Variable(torch.cat(points_batch, 0)).cuda()
    target_v = Variable(torch.cat(targets, 0)).cuda()
    cutdim_processed = []
    for i in range(len(cutdim_batch[0])):
        cutdim_processed.append(torch.stack([item[i] for item in cutdim_batch], 0))

    pred = net(points_v, cutdim_processed)

    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target_v.data).cpu().sum()
    loss = F.nll_loss(pred, target_v)
    if not test:
        loss.backward()
        
    losses.append(loss.data[0])
    
    if not test:
        optimizer.step()
    else:
        sum_correct += correct
        sum_sample += bt
        if sum_sample > 0:
            print("accuracy: %d/%d = %f" % (sum_correct, sum_sample, sum_correct / float(sum_sample)))
            
            
    print('batch: %d, loss: %f, correct %d/%d' %( it, np.mean(losses), correct, bt))

    if it % 1000 == 0:
        torch.save(net.state_dict(), 'mg_model_cuda_%d.pth' % (it))

