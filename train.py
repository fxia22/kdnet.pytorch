from datasets import PartDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class KDNet(nn.Module):
    def __init__(self, k = 16):
        super(KDNet, self).__init__()
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
        self.fc = nn.Linear(1024, k)

    def forward(self, x, c):
        def kdconv(x, dim, featdim, sel, conv):
            x =  F.relu(conv(x))
            x = x.view(-1, featdim, 3, dim)
            x = x.view(-1, featdim, 3 * dim)
            sel = Variable(sel + (torch.arange(0,dim) * 3).long())
            if x.is_cuda:
                sel = sel.cuda()     
            x = torch.index_select(x, dim = 2, index = sel)
            x = x.view(-1, featdim, dim/2, 2)
            x = torch.squeeze(torch.max(x, dim = -1)[0], 3)
            return x      
        
        x1 = kdconv(x, 2048, 8, c[-1], self.conv1)
        x2 = kdconv(x1, 1024, 32, c[-2], self.conv2)
        x3 = kdconv(x2, 512, 64, c[-3], self.conv3)
        x4 = kdconv(x3, 256, 64, c[-4], self.conv4)
        x5 = kdconv(x4, 128, 64, c[-5], self.conv5)
        x6 = kdconv(x5, 64, 128, c[-6], self.conv6)
        x7 = kdconv(x6, 32, 256, c[-7], self.conv7)
        x8 = kdconv(x7, 16, 512, c[-8], self.conv8)
        x9 = kdconv(x8, 8, 512, c[-9], self.conv9)
        x10 = kdconv(x9, 4, 512, c[-10], self.conv10)
        x11 = kdconv(x10, 2, 1024, c[-11], self.conv11)
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
    
    #if torch.numel(left_idx) > 0:
    #    left_idx = left_idx[:,0]
    #if torch.numel(right_idx) > 0:
    #    right_idx = right_idx[:,0]
    #if torch.numel(middle_idx) > 0:
    #    middle_idx = middle_idx[:,0] 
    
    if torch.numel(left_idx) < num_points:
        left_idx = torch.cat([left_idx, middle_idx[0:1].repeat(num_points - torch.numel(left_idx))], 0)
    if torch.numel(right_idx) < num_points:
        right_idx = torch.cat([right_idx, middle_idx[0:1].repeat(num_points - torch.numel(right_idx))], 0)
    
    left_ps = torch.index_select(point_set, dim = 0, index = left_idx)
    right_ps = torch.index_select(point_set, dim = 0, index = right_idx)
    return left_ps, right_ps, dim 

d = PartDataset(root = '../unsupervised3d/shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
l = len(d)
print(len(d.classes), l)
levels = (np.log(2048)/np.log(2)).astype(int)
cutdim = torch.zeros((levels)).long()
net = KDNet().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for it in range(1000):
    optimizer.zero_grad()
    losses = []
    corrects = []
    for batch in range(10):
        j = np.random.randint(l)
        point_set, class_label = d[j]
        target = Variable(class_label).cuda()
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
        cutdim = [(torch.from_numpy(np.array(item).astype(np.int64))) for item in cutdim]
        points = torch.stack(tree[-1])
        points_v = Variable(torch.unsqueeze(torch.squeeze(points), 0)).transpose(2,1).cuda()

        pred = net(points_v, cutdim)
        
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        corrects.append(correct)
        loss = F.nll_loss(pred, target)
        loss.backward()
    
        losses.append(loss.data[0])
        
    optimizer.step()
    print('batch: %d, loss: %f, correct %d/10' %( it, np.mean(losses), np.sum(corrects)))