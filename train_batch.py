from datasets import PartDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class KDNet_Batch(nn.Module):
    def __init__(self, k = 16):
        super(KDNet_Batch, self).__init__()
        self.conv1 = nn.Conv1d(3,8 * 3,1,1)
        self.conv2 = nn.Conv1d(8,32 * 3,1,1)
        self.conv3 = nn.Conv1d(32,64 * 3,1,1)
        self.conv4 = nn.Conv1d(64,64 * 3,1,1)
        self.conv5 = nn.Conv1d(64,64 * 3,1,1)
        self.conv6 = nn.Conv1d(64,128 * 3,1,1)
        self.conv7 = nn.Conv1d(128,256 * 3,1,1)
        self.conv8 = nn.Conv1d(256,512 * 3,1,1)
        self.conv9 = nn.Conv1d(512,512 * 3,1,1)
        self.conv10 = nn.Conv1d(512,512 * 3,1,1)
        self.conv11 = nn.Conv1d(512,1024 * 3,1,1)   
        self.bn1 = nn.BatchNorm1d(8*3)
        self.bn2 = nn.BatchNorm1d(32*3)
        self.bn3 = nn.BatchNorm1d(64*3)
        self.bn4 = nn.BatchNorm1d(64*3)
        self.bn5 = nn.BatchNorm1d(64*3)
        self.bn6 = nn.BatchNorm1d(128*3)
        self.bn7 = nn.BatchNorm1d(256*3)
        self.bn8 = nn.BatchNorm1d(512*3)
        self.bn9 = nn.BatchNorm1d(512*3)
        self.bn10 = nn.BatchNorm1d(512*3)
        self.bn11 = nn.BatchNorm1d(1024*3)
        
        
        self.fc = nn.Linear(1024, k)

    def forward(self, x, c):
        def kdconv(x, dim, featdim, sel, conv, bn):
            batchsize = x.size(0)
            x =  F.relu(bn(conv(x)))
            x = x.view(-1, featdim, 3, dim)
            x = x.view(-1, featdim, 3 * dim)
            x = x.transpose(1,0).contiguous()
            x = x.view(featdim, 3 * dim * batchsize)
            #print x.size()
            sel = Variable(sel + (torch.arange(0,dim) * 3).repeat(batchsize,1).long()).view(-1,1)
            #print sel.size()
            offset = Variable((torch.arange(0,batchsize) * dim * 3).repeat(dim,1).transpose(1,0).contiguous().long().view(-1,1))
            sel = sel+offset
            
            if x.is_cuda:
                sel = sel.cuda()     
            sel = sel.squeeze()
            
            x = torch.index_select(x, dim = 1, index = sel)   
            x = x.view(featdim, batchsize, dim)
            x = x.transpose(1,0).contiguous()
            x = x.view(-1, featdim, dim/2, 2)
            x = torch.squeeze(torch.max(x, dim = -1)[0], 3)
            return x      
        
        x1 = kdconv(x, 2048, 8, c[-1], self.conv1, self.bn1)
        x2 = kdconv(x1, 1024, 32, c[-2], self.conv2, self.bn2)
        x3 = kdconv(x2, 512, 64, c[-3], self.conv3, self.bn3)
        x4 = kdconv(x3, 256, 64, c[-4], self.conv4, self.bn4)
        x5 = kdconv(x4, 128, 64, c[-5], self.conv5, self.bn5)
        x6 = kdconv(x5, 64, 128, c[-6], self.conv6, self.bn6)
        x7 = kdconv(x6, 32, 256, c[-7], self.conv7, self.bn7)
        x8 = kdconv(x7, 16, 512, c[-8], self.conv8, self.bn8)
        x9 = kdconv(x8, 8, 512, c[-9], self.conv9, self.bn9)
        x10 = kdconv(x9, 4, 512, c[-10], self.conv10, self.bn10)
        x11 = kdconv(x10, 2, 1024, c[-11], self.conv11, self.bn11)
        x11 = x11.view(-1,1024)
        out = F.log_softmax(self.fc(x11))
        return out
    
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

d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
l = len(d)
print(len(d.classes), l)
levels = (np.log(2048)/np.log(2)).astype(int)
net = KDNet_Batch().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

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
    loss.backward()
    losses.append(loss.data[0])
    optimizer.step()
    print('batch: %d, loss: %f, correct %d/%d' %( it, np.mean(losses), correct, bt))

    if it % 1000 == 0:
        torch.save(net.state_dict(), 'save_model_%d.pth' % (it))

