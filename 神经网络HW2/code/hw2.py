import numpy as np
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV


train_data = np.load("../data/train_data.npy")
#print(train_data.shape)
#print(train_data)
train_label = np.load("../data/train_label.npy")
#print(train_label.dtype)
train_label = train_label.astype(np.int64)
#print(np.unique(train_label))
#print(train_label.shape)
#print(train_label)
test_data = np.load("../data/test_data.npy")
test_label = np.load("../data/test_label.npy")
#print(test_data.shape,test_label.shape)

def Test():
    clfb = svm.SVC(gamma = 'scale', decision_function_shape='ovr',kernel='rbf');
    grid = GridSearchCV(clfb, param_grid={"C":[0.1, 1, 10, 100]}, cv=3,n_jobs=-1,verbose=2)
    grid.fit(train_data,train_label)
    best_estimator = grid.best_estimator_

    #score = grid.score(test_data,test_label)

    y_train = best_estimator.predict(train_data)
    accurancy_train = np.sum(np.equal(y_train,train_label))/train_label.shape[0]
    
    y_pre = best_estimator.predict(test_data)
    accurancy = np.sum(np.equal(y_pre,test_label))/test_label.shape[0]

    print(y_pre)
    
    print("the best parameters:",grid.best_params_,'\n')
    print(grid.best_score_,'\n')
    print("The third part onevsrest train accurancy and test accurncy:",accurancy_train,accurancy)
    print('\n')
    #clf = svm.SVC(gamma = 'scale',C=1.0,decision_function_shape='ovr',kernel='rbf')
    #clf.fit(train_data,train_label)
    #result = clf.predict()
    
def onevsrest():
#    print(np.unique(train_label))
    train_label1 = train_label.copy()
    train_label1[train_label1!=1] = 2
#    print(train_label1.dtype)
#    print(np.unique(train_label1))
#    print(train_data.shape,train_label.shape,train_label1.shape)
#    print(train_label1)
    train_label2 = train_label.copy()
    train_label2[train_label2!=0] = 2
    train_label3 = train_label.copy()
    train_label3[train_label3!=-1] = 2
 #   train_label1 = train_label1.astype(np.int64)
 #   train_label2 = train_label2.astype(np.int64)
 #   train_label3 = train_label3.astype(np.int64)
 #   print(np.unique(train_label1))
 #   print(train_label1)

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
    accurancy = np.sum(np.equal(yr_pre,test_label))/test_label.shape[0]
    print('The test accurancy:',accurancy,'\n')    
    
    
if __name__ == '__main__':
    onevsrest()
    Test()