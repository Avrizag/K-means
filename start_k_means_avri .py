# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 19:25:32 2021

@author: zivi9
"""

import numpy as np
from time import time

def getData():#fetch data from txt into matrix 
    f = open("data_1_3.txt", "r")
    ret = []
    for l in f:#scan lines
        ret.append([float(i) for i in l.split(",")])#insert words
    return ret
 
def probMaker (dist):
    ret = dist/dist.sum()
    ret[ret<0] = 0
    return ret


def initCenter(k,data):#init centers for kmeans ++
    m,n = data.shape
    firstIndex = np.random.randint(0,m)#select first random index center 
    means = [data[firstIndex,:]]#insert first center
    for i in range(k-1):
        distToMeans = calcdistance(data,np.array(means))#distance between vectors to all the centers 
        dist = distToMeans.min(axis = 1)#find the new index from the new mean 
        probability = probMaker(dist)
        index = np.random.choice(m,1,p=probability)[0]
        means.append(data[index,:]) 
    return np.array(means)#matrix of centers 
        
def calcdistance(a,b):  #calc distance between 2 matrix 
    
  asize = a.shape[0]
  bsize = b.shape[0]
  x = np.sum(a**2, axis=1).reshape([asize,1])#a^2
  y = np.sum(b**2, axis=1).reshape([1,bsize])
  xy = np.dot(a, b.T)
  d = x + y - 2*xy
  d[d<0]=0
 # dist = np.sqrt(d)
    #dist = np.sqrt(np.sum(np.square(a-b)))
   
  return d

    
    
    
    
def findClustForData(means,data):#set for each point the nearst cluster, help : (axis=1)
   
    dist = calcdistance(data,means)
    return dist.argmin(axis=1)    #take the min distace index (cluster)
    # ret = []*len(data)
    # for i in range (len(data)):
    #     min1 = calcdistance(data[i], means[0])
    #     index = 0
    #     for j in range (len(means)):
    #         temp = calcdistance(data[i], means[j])
    #         if (temp < min1):
    #             min1 = temp
    #             index = j
    #     ret.append(index)
    # return ret
                

def calcMeans(data,clust,k):#calc the new center help:np.mean(d,axis=0)
 ret = [0]*k 
 for i in range (k):
       curClust = data[clust == i ,:]    #the data in the same cluster
       ret[i] = np.mean(curClust,axis=0)#find the mean vector of the cluster
               
 # ret = []*len(clust)  
 # for i in range (k):
 #       sum1 = 0
 #       for j in range (len(clust)):
 #           if(clust[j] == i):
 #               sum1+=data[i]
 #       ret[i]=sum1        
               

 return np.array(ret)


def kmeans(data,k):
    means = initCenter(k,data)#array of k dots 
    while True:
        c = findClustForData(means,data)#set for each point the nearst cluster
        newMeans = calcMeans(data,c,k)#calc the new centers
        if (newMeans == means).all():
            return c
        means = newMeans
    

# t= time()
# out = kmeans(d,11)
# print(time()-t)
#ts.PlotData(d,out)

# def sameClusterMean(data,out,pointIndex):
    
#     me = data[pointIndex,:].reshape([1,data.shape[1]])
#     curClust =  data[out == out[pointIndex] ,:]
#     dist = calcdistance(me,curClust)
#     return  dist.mean()
       
# def nearestcluster (data,out,pointIndex,k):
#         ret=[]
#         me = data[pointIndex,:].reshape([1,data.shape[1]])
#         for j in range(k):
#             if j == out[pointIndex]:
#                 continue
#             otherClust = data[ (out==j) ,:]
#             dist = calcdistance(me,otherClust)
#             ret.append(dist.mean())
#         return min(ret)
    
# def calcsiluete(data,out,k,pointIndex):
#     print("sil",pointIndex)
#     a= sameClusterMean(data,out,pointIndex)
#     b = nearestcluster(data, out, pointIndex, k)
#     score = (b-a)/max(a,b)
#     return score

def sillhuete(data,out,k):
    m = data.shape[0]
    means = calcMeans(data,out,k)#calc the new centers
    dist = calcdistance(data,means) #matrix of distances 
    res = np.empty([m])
    for i in range(m):
        myClust = out[i]# cluster of index i 
        row = dist[i,:]#take the distance of point i fron all the means 
        a = row[myClust]#the distance from my mean 
        otherClust = row[np.arange(k)!=myClust]#array of distance from other means 
        b = otherClust.min()#the distance from nearest cluster 
        res[i] = (b-a)/max(a,b)#sillhuete for point 
    return res.mean()

def kmeans_shillhuate(data,max_k =10):
    outs = [kmeans(data,i) for i in range(2,max_k)]#matrix of k means 
    sils = [sillhuete(data,outs[i-2],i)for i in range(2,max_k)]#array of sillhuete score for each kmeans
    ind = np.array(sils).argmax()#return the index of the best score (k )
    return outs[ind],ind+2#return the best k means 



d = getData()
d = np.array(d)
t= time()
out = kmeans(d,10)
#s=sillhuete(d,out,10) 
#ret,index = kmeans_shillhuate(d,15)

print(time()-t)      

from sklearn.cluster  import KMeans 
t= time()
#out = kmeans(d,10)
#s=sillhuete(d,out,10) 
ret = KMeans(n_clusters=10).fit(d)

print(time()-t)    
  
        
    
       
    
    