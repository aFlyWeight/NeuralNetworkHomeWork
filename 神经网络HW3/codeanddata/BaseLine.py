import numpy as np
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
import reading as rd


def BaseLineTrain(data):
    fp = open("log_SVM.txt",'w')
    for item in list(data.keys()):                                  #使用留一法进行训练
        fp.write(item+'\n')       
        data_src, label_src,data_tar,label_tar = rd.data_transfer_svm(data,item)
        data_src = np.array(data_src)
        label_src = np.array(label_src)+1
        data_tar = np.array(data_tar)
        label_tar = np.array(label_tar)+1
        data_size = data_src.shape[0]
        
        clf = svm.SVC(kernel='rbf',probability=True)
        clf.fit(data_src,label_src)
        y_train = clf.predict(data_src)
        accurancy_train = np.sum(np.equal(y_train,label_src))/label_src.shape[0]
        print('The train accurancy:',  accurancy_train,'\n')
        fp.write('The train accurancy: {}\n'.format(accurancy_train))
        
        y_pre = clf.predict(data_tar)
        accurancy_test = np.sum(np.equal(y_pre,label_tar))/label_tar.shape[0]
        print('The test accurancy:',  accurancy_test,'\n')
        fp.write('The test accurancy: {}\n'.format(accurancy_test))
    fp.close()          