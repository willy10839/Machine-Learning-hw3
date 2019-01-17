from PIL import Image
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tqdm import tqdm
import math

Kcluster=5
im = Image.open("hw3.jpg")
image = np.array(im)
img_resize = im.resize((int(image.shape[1]*0.25), int(image.shape[0]*0.25))) 
image_resize = np.array(img_resize)
ary=np.zeros((image_resize.shape[0],image_resize.shape[1]))
means=[]
for i in range(Kcluster):
    tmp=[]
    for j in range(3):
        tmp.append(np.random.uniform(0,255))
    means.append(tmp)

def dist(a,b):
    total=0
    for i in range(len(a)):
        total+=(a[i]-b[i])**2
    return total

def cal(mea,inp):
    qq=[]
    qqq=[]
    for i in range(3):
        qq.append(float(inp[i])-mea[i])
    for i in range(3):
        tmp1=[]
        for j in range(3):
            tmp1.append(qq[i]*qq[j])
        qqq.append(tmp1)
    return np.array(qqq)

def kmeans(k,ary,means,ima):
    pi=[]
    after_means = np.zeros_like(means)
    for i in range(image_resize.shape[0]):
        for j in range(image_resize.shape[1]):
            distance = []
            for z in range(k):
                distance.append(dist(ima[i][j],means[z]))
            ary[i,j] = np.argmin(distance)
    covariance=[]
    for i in range(k):
        cluster = np.where(ary == i)
        pi.append(len(cluster[0])/(image_resize.shape[0]*image_resize.shape[1]))
        total_r=0
        total_g=0
        total_b=0
        if len(cluster[0]) != 0 :
            for j in range(len(cluster[0])):
                total_r+=ima[cluster[0][j],cluster[1][j]][0]
                total_g+=ima[cluster[0][j],cluster[1][j]][1]
                total_b+=ima[cluster[0][j],cluster[1][j]][2]
            after_means[i]=[total_r/len(cluster[0]),total_g/len(cluster[0]),total_b/len(cluster[0])]# i is i-th cluster
        tmp_all=np.zeros((3,3))
        for p in range(image_resize.shape[0]):
            for z in range(image_resize.shape[1]):
                if ary[p][z]==i:
                    tmp_all += cal(after_means[i],ima[p][z])
        covariance.append(tmp_all/len(cluster[0]))
    return after_means,pi,covariance

for i in range(100):
    means,pi,covariance=kmeans(Kcluster,ary,means,image_resize)

print("k-means mean:\n",means)
#np.save("k3means.npy",means)
print("k-means pi: \n",pi)
print("k-means covariace:\n ",covariance)

def normal(mea,co,indata):
    q=[]
    for i in range(3):
        q.append(float(indata[i]))
    tmp=np.array(q)
    for i in range(3):
        tmp[i]=tmp[i]-mea[i]
    out=np.sqrt(((math.pi)**Kcluster)*np.linalg.det(co))*np.exp((-0.5)*tmp.reshape(1,3).dot(inv(co)).dot(tmp.reshape(3,1)))
    return out

def cal_g(indata,mea,co,pi,k):
    total=0
    density=pi[k]*multivariate_normal.pdf(indata, mea[k], co[k])
    for i in range(Kcluster):
        total+=pi[i]*multivariate_normal.pdf(indata, mea[i], co[i])
    return float(density/total)

def recal_mean(image,mean,co,pi):
    new_means=[]
    for i in range(Kcluster):
        tmp_means=[]
        N_k=0
        r=0
        g=0
        b=0
        for j in range(image_resize.shape[0]):
            for z in range(image_resize.shape[1]):
                tmp = cal_g(image[j][z],mean,co,pi,i)
                N_k+=tmp
                r+=image[j][z][0]*tmp
                g+=image[j][z][1]*tmp
                b+=image[j][z][2]*tmp
        tmp_means.append(r/N_k)
        tmp_means.append(g/N_k)  
        tmp_means.append(b/N_k)
        new_means.append(tmp_means)
    return new_means

def recal_pi(imag,mean,co,pi):
    new_pi=[]
    for i in range(Kcluster):
        N_k=0
        for j in range(image_resize.shape[0]):
            for z in range(image_resize.shape[1]):
                tmp = cal_g(imag[j][z],mean,co,pi,i)
                N_k+=tmp
        new_pi.append(N_k/(image_resize.shape[0]*image_resize.shape[1]))
    return new_pi

def recal_co(imag,new_mean,co,pi,old_mean):
    new_co=np.zeros((Kcluster,3,3))
    for i in range(Kcluster):
        N_k=0
        tmp_co=np.zeros((3,3))
        for j in range(image_resize.shape[0]):
            for z in range(image_resize.shape[1]):
                tmp = cal_g(imag[j][z],old_mean,co,pi,i)
                N_k+=tmp
                tmp_co += tmp*cal(new_mean[i],imag[j][z])
        new_co[i]=tmp_co/N_k
    return new_co

def cal_likelihood(mean,co,pi):
    out=0.0
    for i in range(image_resize.shape[0]):
        for j in range(image_resize.shape[1]):
            tmp=0.0
            for z in range(Kcluster):
                tmp+=pi[z]*multivariate_normal.pdf(image_resize[i][j], mean[z], co[z])
            out+=np.log(tmp)
    return float(out)

def recal_var(mean,co,pi,imag):
    new_means=recal_mean(imag,mean,co,pi)
    new_co=recal_co(imag,new_means,co,pi,mean)
    new_pi=recal_pi(imag,mean,co,pi)
    return new_means,new_co,new_pi

def construct(mean,co,pi,ima):
    new_image = np.zeros_like(ima)
    for i in range(image_resize.shape[0]):
        for j in range(image_resize.shape[1]):
            tmp=[]
            for k in range(Kcluster):
                tmp.append(cal_g(ima[i][j],mean,co,pi,k))
            idx = np.argmax(tmp)
            new_image[i][j][0] = mean[idx][0]
            new_image[i][j][1] = mean[idx][1]
            new_image[i][j][2] = mean[idx][2]
    return  new_image


new_m=means
new_c=covariance
new_p=pi
likeli=[]
for i in tqdm(range(100)):
    new_m,new_c,new_p=recal_var(new_m,new_c,new_p,image_resize)
    likeli.append(cal_likelihood(new_m,new_c,new_p))

plt.plot(likeli, 'r')
plt.title('log likelihood')
plt.figure()
outimag=construct(new_m,new_c,new_p,image_resize)
plt.imshow(outimag)
#np.save("k20means.npy",means)
np.save("k3means_image.npy",outimag)
np.save("k3means_like.npy",likeli)
