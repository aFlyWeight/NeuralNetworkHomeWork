import reading as rd
from DANN import DANN
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import tqdm
import numpy as np
import BaseLine as BL

torch.cuda._initialized = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)


def train(data):
    epoch_num = 2000
    lr = 0.001
    fp = open("log.txt",'w')
    for item in list(data.keys()):                                  #使用留一法进行训练
        best_acc = -float('inf')
        model = DANN(310,128,128,64,64,3,64,64,2).to(device)
        loss_domain = nn.CrossEntropyLoss()
        loss_class = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),lr=lr)
        
        data_src_domain, label_src_domain,data_tar_domain,label_tar_domain = rd.data_transfer(data,item)
        data_src_domain = torch.from_numpy(np.array(data_src_domain)).float().to(device)
        label_src_domain = torch.from_numpy(np.array(label_src_domain)+1).long().to(device)
        data_tar_domain = torch.from_numpy(np.array(data_tar_domain)).float().to(device)
        label_tar_domain = np.array(label_tar_domain)+1
        data_size = data_src_domain.shape[0]
#        print(data_size)
#        print(data_src_domain.shape,label_src_domain.shape)
        fp.write(item+'\n')
        
        for ep in range(epoch_num):
            model.train()
            p = float(ep)/epoch_num
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
 #           lr = float(lr)/pow(1+10*p,0.75)
            #训练源域
            domain_src_label = torch.ones(data_size).long().to(device)        #域标签
 #           print(domain_src_label.shape)
            class_output, domain_src_output = model(input_data=data_src_domain, alpha=alpha)
#            print(class_output.shape) 
#            print(class_output)
#            print(domain_src_output.shape)
#            print(domain_src_output)
            err_s_label = loss_class(class_output, label_src_domain) 
            err_s_domain = loss_domain(domain_src_output,domain_src_label)
            #训练目标域
            domain_tar_label = torch.zeros(data_size).long().to(device)
            label_tar_output,domain_tar_output = model(input_data=data_tar_domain, alpha=alpha)
            err_t_domain = loss_domain(domain_tar_output,domain_tar_label)
            err_domain = err_t_domain + err_s_domain
            err = err_s_label + err_domain
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            p_label_tar_output = torch.argmax(label_tar_output,dim=1).cpu().numpy()
 #           print(p_label_tar_output)
 #           print(np.sum(p_label_tar_output == 2))
 #           print(label_tar_domain)
            acc_loss = np.sum(label_tar_domain == p_label_tar_output) / data_size
            if(best_acc < acc_loss):
                best_acc = acc_loss
            item_pr = 'Epoch: [{}/{}], classify_loss: {:.4f}, domain_loss_s: {:.4f}, domain_loss_t: {:.4f}, domain_loss: {:.4f},total_loss: {:.4f}, accurance: {:.4f}'.format(
                ep, epoch_num, err_s_label.item(), err_s_domain.item(), err_t_domain.item(), err_domain.item(), err.item(), acc_loss)
            print(item_pr)
            fp.write(item_pr + '\n')
        print('best_acc: ' ,best_acc, '\n')
        fp.write('best_acc: {}\n'.format(best_acc))
    fp.close()
        
            
          
if __name__ == '__main__':
    data = rd.load_data()
    train(data)
 #   BL.BaseLineTrain(data)