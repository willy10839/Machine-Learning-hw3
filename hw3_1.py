import numpy as np
from numpy.linalg import pinv,inv,matrix_power
from scipy.linalg import sqrtm
from scipy.io import loadmat
import csv
import matplotlib.pyplot as plt
train=[]
test=[]
data=[]
sqr_exp_kernel = [1.0,4.0,0.0,0.0]
linear_kernel = [0.0,0.0,0.0,1.0]
exp_quadra_kernel = [1.0,4.0,0.0,5.0]
exp_quadra_kernel_2 = [1.0,64.0,10.0,0.0]
with open('gp.csv', newline='') as csvfile:
    train_data = csv.reader(csvfile)
    for i in  train_data:
        i = [float(j) for j in i]
        data.append(i)
for i in range(0,60):
    train.append(data[i])
for i in range(60,120):
    test.append(data[i])
train_data=sorted(train, key=lambda k: k[0], reverse=False)
test_data=sorted(test, key=lambda k: k[0], reverse=False)

def k(kernel,i,j):
    return kernel[0]*np.exp(-kernel[1]/2*(i-j)*(i-j))+kernel[2]+kernel[3]*i*j

def cal(kernel,train_input):
    target=[]
    indata=[]
    test_target=[]
    for i in train_input:
        target.append(i[1])
        indata.append(i[0])
        test_target.append(i[1])
    iden=np.identity(len(train_input))
    covariance=[]
    k_value=[]
    c=[]
    variance=[]
    for i in range(len(train_input)):
        tmp=[]
        tmp1=[]
        c.append(k(kernel,train_input[i][0],train_input[i][0])+1)
        for j in range(len(train_input)):
            tmp.append(k(kernel,train_input[i][0],train_input[j][0])+iden[i][j])
            tmp1.append(k(kernel,train_input[i][0],train_input[j][0]))
        covariance.append(tmp)
        k_value.append(tmp1)
    co_mat=np.array(covariance)
    k_mat=np.array(k_value)
    mean=k_mat.dot(inv(co_mat)).dot(target)
    for i in range(len(train_input)):
        variance.append(c[i]-k_mat[i].dot(inv(co_mat)).dot(k_mat[i].T))
    sq=np.sqrt(variance)
    plt.fill_between(indata,mean-sq,mean+sq,facecolor='#ffa5d2')
    plt.plot(indata, mean, 'r--')
    plt.plot(indata,target,'bo')
    plt.title(str(kernel))
    plt.show()
    
    c_test=[]
    k_test=[]
    for i in range(len(train_input)):
        c_test.append(k(kernel,test_data[i][0],test_data[i][0])+1)
    for i in range(len(train_input)):
        tmp2=[]
        for i in range(len(train_input)):
            tmp2.append(k(kernel,test_data[i][0],test_data[j][0])+iden[i][j])
        k_test.append(tmp2)
    k_test_mat=np.array(k_test)
    mean_test=k_test_mat.dot(inv(co_mat)).dot(test_target)
    rms=0
    rms_test=0
    for i in range(len(train_input)):
        rms+=(mean[i]-target[i])*(mean[i]-target[i])
        rms_test+=(mean_test[i]-test_target[i])*(mean_test[i]-test_target[i])
    #print(rms)
    #print(rms_test)
    print(np.sqrt(rms/len(train_input)))
    print(np.sqrt(rms_test/len(train_input)))
cal(sqr_exp_kernel,train_data)
cal(linear_kernel,train_data)
cal(exp_quadra_kernel,train_data)
cal(exp_quadra_kernel_2,train_data)
