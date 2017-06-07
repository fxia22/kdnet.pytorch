import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class KDNet_Batch(nn.Module):
    def __init__(self, k = 16):
        super(KDNet_Batch, self).__init__()
        self.conv1 = nn.Conv1d(3,8 * 3,1,1)
        self.conv2 = nn.Conv1d(8*2,32 * 3,1,1)
        self.conv3 = nn.Conv1d(32*2,64 * 3,1,1)
        self.conv4 = nn.Conv1d(64*2,64 * 3,1,1)
        self.conv5 = nn.Conv1d(64*2,64 * 3,1,1)
        self.conv6 = nn.Conv1d(64*2,128 * 3,1,1)
        self.conv7 = nn.Conv1d(128*2,256 * 3,1,1)
        self.conv8 = nn.Conv1d(256*2,512 * 3,1,1)
        self.conv9 = nn.Conv1d(512*2,512 * 3,1,1)
        self.conv10 = nn.Conv1d(512*2,512 * 3,1,1)
        self.conv11 = nn.Conv1d(512*2,1024 * 3,1,1)   
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
        
        
        self.fc = nn.Linear(1024 * 2, k)

    def forward(self, x, c):
        def kdconv(x, dim, featdim, sel, conv, bn):
            batchsize = x.size(0)
            #print(x.size())
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
            x = x.transpose(2,1).contiguous() 
            x = x.view(-1, dim/2, featdim * 2)     
            x = x.transpose(2,1).contiguous()
            #x = x.view(-1, featdim, dim/2, 2)
            #x = torch.squeeze(torch.max(x, dim = -1)[0], 3)  
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
        x11 = x11.view(-1,1024 * 2)
        out = F.log_softmax(self.fc(x11))
        return out
    
    
    
class KDNet_Batch_mp_4d(nn.Module):
    def __init__(self, k = 16):
        super(KDNet_Batch, self).__init__()
        self.conv1 = nn.Conv1d(4,8 * 3,1,1)
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
    
    
class KDNet_Batch_mp(nn.Module):
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