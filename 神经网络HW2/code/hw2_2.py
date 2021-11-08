import numpy as np
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
import random
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

train_data = np.load("../data/train_data.npy")
#print(train_data.shape)
#print(train_data)
train_label = np.load("../data/train_label.npy")
#print(train_label.dtype)
train_label = train_label.astype(np.int64)
#print(np.unique(train_label))
print(train_label.shape)
#print(train_label)
test_data = np.load("../data/test_data.npy")
test_label = np.load("../data/test_label.npy")
test_label = test_label.astype(np.int64)
#print(test_data.shape,test_label.shape)

#train_labels = train_label.copy()
from collections import Counter
print(Counter(train_label))
#print(train_data.shape)
#train_datas = train_data.copy()
#rnumber = random.randint(0,1)
#print(train_label[10000])
#print(rnumber)

def trainsvm(train_data,train_label):
    clf = svm.SVC(kernel='rbf',probability=True)
    clf.fit(train_data,train_label)
    #重新跑一边训练集，获取训练精度
    y_train = clf.predict(train_data)
    accurancy_train = np.sum(np.equal(y_train,train_label))/train_label.shape[0]
    print('The train accurancy:',  accurancy_train,'\n')
    #进行预测，并获取预测误差
    y_pre = clf.predict(test_data)
    y_pre_pro = clf.predict_proba(test_data)
#    print('-----------------------------')
#    print(y_pre)
    accurancy = np.sum(np.equal(y_pre,test_label))/test_label.shape[0]
#    print('The test accurancy:',accurancy,'\n')    
    return y_pre, y_pre_pro
'''
def plot_roc_curve(target_label,pre_label):
    fpr,tpr,thresholds = roc_curve(target_label,pre_label)
    plt.figure(figsize = (10,10))
    plt.xlim(0,1)
    plt.ylim(0.0,1.1)
    plt.xlabel('False Postive Rate')
    plt.ylabel('True Postive Rate')
    plt.plot(fpr,tpr)
    plt.show()
'''    

def SampleDivision():
    #进性onevsrest分类,正类标签为1
    train_label11 = []
    train_data11 = []
    #之后用先验知识进行分类，比如1和0，1和-1或随机进行分类
    train_label12p1 = []
    train_data12p1 = []
    train_label12p2 = []
    train_data12p2 = []
    train_label12r1 = []
    train_data12r1 = []
    train_label12r2 = []
    train_data12r2 = []
    for i in range(0,train_label.shape[0]):
        if train_label[i] == 1:
            train_label11.append(train_label[i])
            train_data11.append(train_data[i])
        else:
            #分别用先验和随机对其余的数据进行分类
            if train_label[i] == -1:
                train_label12p1.append(2)
                train_data12p1.append(train_data[i])
            else:
                train_label12p2.append(2)
                train_data12p2.append(train_data[i])
            #随机对样本进行分类
            rnumber = random.randint(1,2)
            if rnumber == 1:
                train_label12r1.append(2)
                train_data12r1.append(train_data[i])
            else:
                train_label12r2.append(2)
                train_data12r2.append(train_data[i])
 #   print(np.array(train_label12r1).shape,np.array(train_data12r1).shape)
    train_label11 = np.array(train_label11)
    train_data11 = np.array(train_data11)
    train_label12p1 = np.array(train_label12p1)
    train_data12p1 = np.array(train_data12p1)
    train_label12p2 = np.array(train_label12p2)
    train_data12p2 = np.array(train_data12p2)
    train_label12r1 = np.array(train_label12r1)
    train_data12r1 = np.array(train_data12r1)
    train_label12r2 = np.array(train_label12r2)
    train_data12r2 = np.array(train_data12r2)
    
    #将分裂的数据分别进行合并
    train_label1p1 = np.concatenate((train_label11,train_label12p1))
    train_data1p1 = np.concatenate((train_data11,train_data12p1))
    train_label1p2 = np.concatenate((train_label11,train_label12p2))
    train_data1p2 = np.concatenate((train_data11,train_data12p2))
    train_label1r1 = np.concatenate((train_label11,train_label12r1))
    train_data1r1 = np.concatenate((train_data11,train_data12r1))
    train_label1r2 = np.concatenate((train_label11,train_label12r2))
    train_data1r2 = np.concatenate((train_data11,train_data12r2))
    #print(train_label11.shape,train_data11.shape)
    
    #进性onevsrest分类,正类标签为0
    train_label21 = []
    train_data21 = []
    #之后用先验知识进行分类，比如1和0，1和-1或随机进行分类
    train_label22p1 = []
    train_data22p1 = []
    train_label22p2 = []
    train_data22p2 = []
    train_label22r1 = []
    train_data22r1 = []
    train_label22r2 = []
    train_data22r2 = []
    for i in range(0,train_label.shape[0]):
        if train_label[i] == 0:
            train_label21.append(train_label[i])
            train_data21.append(train_data[i])
        else:
            #分别用先验和随机对其余的数据进行分类
            if train_label[i] == 1:
                train_label22p1.append(2)
                train_data22p1.append(train_data[i])
            else:
                train_label22p2.append(2)
                train_data22p2.append(train_data[i])
            #随机对样本进行分类
            rnumber = random.randint(1,2)
            if rnumber == 1:
                train_label22r1.append(2)
                train_data22r1.append(train_data[i])
            else:
                train_label22r2.append(2)
                train_data22r2.append(train_data[i])
 #   print(np.array(train_label22r1).shape,np.array(train_data22r1).shape)
    train_label21 = np.array(train_label21)
    train_data21 = np.array(train_data21)
    train_label22p1 = np.array(train_label22p1)
    train_data22p1 = np.array(train_data22p1)
    train_label22p2 = np.array(train_label22p2)
    train_data22p2 = np.array(train_data22p2)
    train_label22r1 = np.array(train_label22r1)
    train_data22r1 = np.array(train_data22r1)
    train_label22r2 = np.array(train_label22r2)
    train_data22r2 = np.array(train_data22r2)
    
    #将分裂的数据分别进行合并
    train_label2p1 = np.concatenate((train_label21,train_label22p1))
    train_data2p1 = np.concatenate((train_data21,train_data22p1))
    train_label2p2 = np.concatenate((train_label21,train_label22p2))
    train_data2p2 = np.concatenate((train_data21,train_data22p2))
    train_label2r1 = np.concatenate((train_label21,train_label22r1))
    train_data2r1 = np.concatenate((train_data21,train_data22r1))
    train_label2r2 = np.concatenate((train_label21,train_label22r2))
    train_data2r2 = np.concatenate((train_data21,train_data22r2))
        
    #进性onevsrest分类,正类标签为-1
    train_label31 = []
    train_data31 = []
    #之后用先验知识进行分类，比如1和0，1和-1或随机进行分类
    train_label32p1 = []
    train_data32p1 = []
    train_label32p2 = []
    train_data32p2 = []
    train_label32r1 = []
    train_data32r1 = []
    train_label32r2 = []
    train_data32r2 = []
    for i in range(0,train_label.shape[0]):
        if train_label[i] == -1:
            train_label31.append(train_label[i])
            train_data31.append(train_data[i])
        else:
            #分别用先验和随机对其余的数据进行分类
            if train_label[i] == 1:
                train_label32p1.append(2)
                train_data32p1.append(train_data[i])
            else:
                train_label32p2.append(2)
                train_data32p2.append(train_data[i])
            #随机对样本进行分类
            rnumber = random.randint(1,2)
            if rnumber == 1:
                train_label32r1.append(2)
                train_data32r1.append(train_data[i])
            else:
                train_label32r2.append(2)
                train_data32r2.append(train_data[i])
 #   print(np.array(train_label32r1).shape,np.array(train_data32r1).shape)
 #   print(train_data32p1)
    train_label31 = np.array(train_label31)
    train_data31 = np.array(train_data31)
    train_label32p1 = np.array(train_label32p1)
    train_data32p1 = np.array(train_data32p1)
 #   print(train_data32p1.shape)
    train_label32p2 = np.array(train_label32p2)
    train_data32p2 = np.array(train_data32p2)
    train_label32r1 = np.array(train_label32r1)
    train_data32r1 = np.array(train_data32r1)
    train_label32r2 = np.array(train_label32r2)
    train_data32r2 = np.array(train_data32r2)
 #   print('#######')
 #   print(train_data31.shape,train_data32p2.shape)
    
    #将分裂的数据分别进行合并
    train_label3p1 = np.concatenate((train_label31,train_label32p1))
    train_data3p1 = np.concatenate((train_data31,train_data32p1))
    train_label3p2 = np.concatenate((train_label31,train_label32p2))
    train_data3p2 = np.concatenate((train_data31,train_data32p2))
    train_label3r1 = np.concatenate((train_label31,train_label32r1))
    train_data3r1 = np.concatenate((train_data31,train_data32r1))
    train_label3r2 = np.concatenate((train_label31,train_label32r2))
    train_data3r2 = np.concatenate((train_data31,train_data32r2))
    
    #先对先验知识分类的样本进行训练，后面一样
    print('sample1p1:\n')
    y_1p1,y_1p1_pro = trainsvm(train_data1p1 , train_label1p1)
    print(y_1p1_pro.shape)
    print('sample1p2:\n')
    y_1p2,y_1p2_pro = trainsvm(train_data1p2 , train_label1p2)
    #进行最小化合并，后面一样
    y_1pmin = y_1p1.copy()
    y_1pmin_pro = np.zeros(y_1p1.shape[0])
    for i in range(0,y_1p1.shape[0]):
        if y_1p1[i] == 1 and y_1p2[i] == 1:
            y_1pmin[i] = 1
            y_1pmin_pro[i] = min(y_1p1_pro[i,0],y_1p2_pro[i,0])
        else:
            y_1pmin[i] = 2
            y_1pmin_pro[i] = 0
    print('sample2p1:\n')
    y_2p1,y_2p1_pro = trainsvm(train_data2p1 , train_label2p1)
    print('sample2p2:\n')
    y_2p2,y_2p2_pro = trainsvm(train_data2p2 , train_label2p2)
    y_2pmin = y_2p1.copy()
    y_2pmin_pro = np.zeros(y_2p1.shape[0])
    for i in range(0,y_2p1.shape[0]):
        if y_2p1[i] == 0 and y_2p2[i] == 0:
            y_2pmin[i] = 0
            y_2pmin_pro[i] = min(y_2p1_pro[i,0],y_2p2_pro[i,0])
        else:
            y_2pmin[i] = 2
            y_2pmin_pro[i] = 0
    print('sample3p1:\n')
    y_3p1,y_3p1_pro = trainsvm(train_data3p1 , train_label3p1)
    print('sample3p2:\n')
    y_3p2,y_3p2_pro = trainsvm(train_data3p2 , train_label3p2)
    y_3pmin = y_3p1.copy()
    y_3pmin_pro = np.zeros(y_3p1.shape[0])
    for i in range(0,y_3p1.shape[0]):
        if y_3p1[i] == -1 and y_3p2[i] == -1:
            y_3pmin[i] = -1
            y_3pmin_pro[i] = min(y_3p1_pro[i,0],y_3p2_pro[i,0])
        else:
            y_3pmin[i] = 2
            y_3pmin_pro[i] = 0
    
    y_pmax = y_1pmin.copy()
    for i in range(0,y_pmax.shape[0]):
        if y_1pmin_pro[i] >= y_2pmin_pro[i] and y_1pmin_pro[i] >= y_3pmin_pro[i]:
            y_pmax[i] = 1
        elif y_2pmin_pro[i] >= y_1pmin_pro[i] and y_2pmin_pro[i] >= y_3pmin_pro[i]:
            y_pmax[i] = 0
        elif y_3pmin_pro[i] >= y_1pmin_pro[i] and y_3pmin_pro[i] >= y_2pmin_pro[i]:
            y_pmax[i] = -1
    
    print(np.unique(y_pmax))
    print('the prio result of minmax:',y_pmax,'\n')
    accurancy_pminmax = np.sum(np.equal(y_pmax,test_label))/test_label.shape[0]
    print('The test accurancy of prior minmax:',accurancy_pminmax,'\n')
    
    #然后对随机的数据进行训练，并进行minmax合并
    print('sample1r1:\n')
    y_1r1,y_1r1_pro = trainsvm(train_data1r1 , train_label1r1)
    print('sample1r2:\n')
    y_1r2,y_1r2_pro = trainsvm(train_data1r2 , train_label1r2)
    #进行最小化合并，后面一样
    y_1rmin = y_1r1.copy()
    y_1rmin_pro = np.zeros(y_1rmin.shape[0])
    for i in range(0,y_1r1.shape[0]):
        if y_1r1[i] == 1 and y_1r2[i] == 1:
            y_1rmin[i] = 1
            y_1rmin_pro[i] = min(y_1r1_pro[i,0],y_1r2_pro[i,0])
        else:
            y_1rmin[i] = 2
            y_1rmin_pro[i] = 0
    print('sample2r1:\n')
    y_2r1,y_2r1_pro = trainsvm(train_data2r1 , train_label2r1)
    print('sample2r2:\n')
    y_2r2,y_2r2_pro = trainsvm(train_data2r2 , train_label2r2)
    y_2rmin = y_2r1.copy()
    y_2rmin_pro = np.zeros(y_2rmin.shape[0])
    for i in range(0,y_2r1.shape[0]):
        if y_2r1[i] == 0 and y_2r2[i] == 0:
            y_2rmin[i] = 0
            y_2rmin_pro[i] = min(y_2r1_pro[i,0],y_2r2_pro[i,0])
        else:
            y_2rmin[i] = 2
            y_2rmin_pro[i] = 0
    print('sample3r1:\n')
    y_3r1,y_3r1_pro = trainsvm(train_data3r1 , train_label3r1)
    print('sample3r2:\n')
    y_3r2,y_3r2_pro = trainsvm(train_data3r2 , train_label3r2)
    y_3rmin = y_3r1.copy()
    y_3rmin_pro = np.zeros(y_3rmin.shape[0])
    for i in range(0,y_3r1.shape[0]):
        if y_3r1[i] == -1 and y_3r2[i] == -1:
            y_3rmin_pro[i] = min(y_3r1_pro[i,0],y_3r2_pro[i,0])
            y_3rmin[i] = -1
        else:
            y_3rmin[i] = 2
            y_3rmin_pro[i] = 0
    
    y_rmax = y_1rmin.copy()
    for i in range(0,y_rmax.shape[0]):
        if y_1rmin_pro[i] >= y_2rmin_pro[i] and y_1rmin_pro[i] >= y_3rmin_pro[i]:
            y_rmax[i] = 1
        elif y_2rmin_pro[i] >= y_1rmin_pro[i] and y_2rmin_pro[i] >= y_3rmin_pro[i]:
            y_rmax[i] = 0
        elif y_3rmin_pro[i] >= y_1rmin_pro[i] and y_3rmin_pro[i] >= y_2rmin_pro[i]:
            y_rmax[i] = -1
    print(np.unique(y_rmax))
    print('the random result of minmax:',y_rmax,'\n')
    accurancy_rminmax = np.sum(np.equal(y_rmax,test_label))/test_label.shape[0]
    print('The test accurancy of random minmax:',accurancy_rminmax,'\n')
    
            
            
    
    '''
    train_label1 = train_label.copy()
    train_label1[train_label1!=1] = 2
    train_label2 = train_label.copy()
    train_label2[train_label2!=0] = 2
    train_label3 = train_label.copy()
    train_label3[train_label3!=-1] = 2

    #分别用三个label训练三个svm
    clf1 = svm.SVC(kernel='rbf')
    clf2 = svm.SVC(kernel='rbf')
    clf3 = svm.SVC(kernel='rbf')
    clf1.fit(train_data,train_label1)
    clf2.fit(train_data,train_label2)
    clf3.fit(train_data,train_label3)
    #重新跑一边训练集，获取训练精度
    y1_train = clf1.predict(train_data)
    y2_train = clf2.predict(train_data)
    y3_train = clf3.predict(train_data)
    y1_train[y1_train == 2] = 0
    y2_train[y2_train == 2] = 0
    y3_train[y3_train == 2] = 0
    yr_train = y1_train + y2_train + y3_train
    accurancy_train = np.sum(np.equal(yr_train,train_label))/train_label.shape[0]
    print('The train accurancy:',  accurancy_train,'\n')
    #进行预测，并获取预测误差
    y1_pre = clf1.predict(test_data)
    y2_pre = clf2.predict(test_data)
    y3_pre = clf3.predict(test_data)
    print('-----------------------------\n')
    print(y1_pre,'\n',y2_pre,'\n',y3_pre)
    y1_pre[y1_pre == 2] = 0
    y2_pre[y2_pre == 2] = 0
    y3_pre[y3_pre == 2] = 0
    yr_pre = y1_pre + y2_pre + y3_pre
    print('\n',yr_pre)
    '''


if __name__ == '__main__':
    SampleDivision()