import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt

df = pd.read_excel('statistic.xlsx')
print(df)

acc_best = df.loc[0:4] 
acc_final = df.loc[5:9]
print(acc_best)
print(acc_final)

newcolumn = ['SVM','Adam_lr = 0.0001_epoch=1000_best','Adam_lr=0.0001_epoch=500_best','Adam_lr=0.001_epoch=1000_best','SGD_lr=0.001_epoch = 1000_best']

acc_best.columns = newcolumn
print(acc_best)

acc_all = pd.merge(acc_final,acc_best)
print(acc_all)
acc_all[['Adam_lr = 0.0001_epoch=1000_best','Adam_lr = 0.0001_epoch=1000']].plot()
plt.xlabel('sub') 
plt.ylabel('accurancy') 
plt.title('Adam_lr = 0.0001_epoch=1000') 

acc_all[['Adam_lr=0.0001_epoch=500_best','Adam_lr=0.0001_epoch=500']].plot()
plt.xlabel('sub') 
plt.ylabel('accurancy') 
plt.title('Adam_lr = 0.0001_epoch=500') 

acc_all[['Adam_lr=0.001_epoch=1000_best','Adam_lr=0.001_epoch=1000']].plot()
plt.xlabel('sub') 
plt.ylabel('accurancy') 
plt.title('Adam_lr = 0.001_epoch=1000') 


acc_all[['SGD_lr=0.001_epoch = 1000_best','SGD_lr=0.001_epoch = 1000']].plot()
plt.xlabel('sub') 
plt.ylabel('accurancy') 
plt.title('SGD_lr=0.001_epoch = 1000') 



acc_final.plot()
plt.xlabel('sub') 
plt.ylabel('accurancy') 
plt.title('Actuall result contrast') 

acc_all.plot()
plt.xlabel('sub') 
plt.ylabel('accurancy') 
plt.title('all result contrast') 
plt.show()
plt.show()

