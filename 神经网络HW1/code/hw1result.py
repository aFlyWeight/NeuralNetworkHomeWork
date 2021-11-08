import matplotlib.pyplot as plt
import numpy as np

t_data = np.loadtxt('./out0.txt')
print(t_data.shape,t_data.dtype)

'''
max_t_error = np.max(t_data,axis=0)
print(max_t_error)
t_data_p = np.where(max_t_error[2] == t_data[:,2])
print(t_data_p)
'''
i = 0
max_t_error = []
max_mr = 0
while i < t_data.shape[0] :
    if max_mr < t_data[i,2]:
        max_mr = t_data[i,2]
        max_t_error.append(t_data[i,:])
    i+=1
print(max_t_error,max_mr)

np.savetxt('parameter.txt',max_t_error)
      
print('------max_t_error:{}--------'.format(max_t_error))
print('\n')
  
'''       
t_data_H = np.where(max_t_error[1] == t_data[:,1])
t_data_l = np.where(max_t_error[0] == t_data[:,0])

print(t_data_H)
print('\n')
print('---------------','\n')
print(t_data_l)
'''  

#print(t_data)
f_data = t_data
i=0
while i < f_data.shape[0] :
    if abs(f_data[i,1]-1/10) <= 1e-8:
        f_data[i,1] = 1
    elif abs(f_data[i,1]-1/100) <= 1e-8:
        f_data[i,1] = 2
    elif  abs(f_data[i,1]-1/1000) <= 1e-8:
        f_data[i,1] = 3
    elif  abs(f_data[i,1]-1/10000) <= 1e-8:
        f_data[i,1] = 4
    elif  abs(f_data[i,1]-1/100000) <= 1e-8:
        f_data[i,1] = 5
    elif  abs(f_data[i,1]-1/1000000) <= 1e-8:
        f_data[i,1] = 6
    elif  abs(f_data[i,1]-1/10000000) <= 1e-8:
        f_data[i,1] = 7     
    i+=1
print(f_data[:,1] )
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(f_data[:,0],f_data[:,1],f_data[:,2],alpha=0.3)
ax.set_xlabel('Hidden num')
ax.set_ylabel('Learning rate')
ax.set_zlabel('Matching rate')
plt.show()
