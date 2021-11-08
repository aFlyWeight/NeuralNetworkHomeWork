import scipy.io as sio
import numpy as np
import torch

test_data_tmp = sio.loadmat("../train_test/test_data")
test_label_tmp = sio.loadmat("../train_test/test_label")
train_data_tmp = sio.loadmat("../train_test/train_data")
train_label_tmp = sio.loadmat("../train_test/train_label")
#print(train_data_tmp['train_data'].dtype)

test_data = np.array(test_data_tmp['test_data'])
test_label = np.array(test_label_tmp['test_label'])+1
train_data = np.array(train_data_tmp['train_data'])
train_label = np.array(train_label_tmp['train_label'])+1
train_label = train_label.reshape(499)
test_label = test_label.reshape(343)

#print(test_label,train_label)

#print(test_data.shape,test_label.shape,train_data.shape,train_label.shape)

torch.cuda._initialized = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

train_data_pytorch = torch.from_numpy(train_data).float().to(device)
train_label_pytorch = torch.from_numpy(train_label).long().to(device)

#print(train_data_pytorch)
#print(train_label_pytorch)

import torch.nn as nn
from torch.nn import functional as F

class hw1net(nn.Module):
    def __init__(self, D_in, Hidden, D_out):
        super(hw1net,self).__init__()
        self.linear_1 = nn.Linear(D_in,Hidden)
        self.linear_2 = nn.Linear(Hidden,D_out)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear_1(x))
#        x = torch.sigmoid(self.linear_2(x))
        x = self.linear_2(x)
#        x = F.softmax(x,dim=0)
        return x

import torch.optim as optim


def train_process(lrtmp):
    criterion = nn.CrossEntropyLoss()
    lr = lrtmp
    optimizer = optim.SGD(model.parameters(), lr=lr)
    epoch_num = 500
    for ep in range(epoch_num):
        model.train(train_data_pytorch)
        output = model(train_data_pytorch)
        loss = criterion(output,train_label_pytorch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (ep+1)%100 == 0:
            model.eval()
            p_out = model(train_data_pytorch)
            p_label = torch.argmax(p_out,dim=1).cpu().numpy()
            acc_loss = np.sum(p_label == train_label) / 500
            print('epoch num : {} -- Loss : {} -- Acc : {}'.format(ep+1,loss.data, acc_loss))
            
test_data_pytorch = torch.from_numpy(test_data).float().to(device)
test_label_pytorch = torch.from_numpy(test_label).long().to(device)

t_data = []
Hidden_num = 100
while Hidden_num <= 3000:
    i=10
    while i <= 1000000:
        model = hw1net(310, Hidden_num, 3).to(device)
        lrtmp = 1/i
        train_process(lrtmp)
        model.eval()
        t_output = model(test_data_pytorch)
        ptest_label = torch.argmax(t_output,1).cpu().numpy()
        t_output=t_output.cpu().detach().numpy()
        #print('{}{}----'.format(t_output,'\n'))
        #print('{}{}----'.format(ptest_label,'\n'))
        test_num = 343
        t_errors = np.sum(ptest_label == test_label)/343
        #print(t_errors)
        t_data.append([Hidden_num,lrtmp,t_errors])
        print('Hidden_num:{}-----------lr:{}------------t_errors:{}'.format(Hidden_num,lrtmp,t_errors))
        i=i*10
    Hidden_num+=5
t_data = np.array(t_data)
np.savetxt('out0.txt',t_data)



            
        
    
        

