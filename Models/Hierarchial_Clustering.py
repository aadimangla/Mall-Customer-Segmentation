#%reset -f
"""
Created on Fri Dec  6 13:05:49 2019

@author: USER
"""
# Hierarchial Clustering

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Using the Dendograms to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendograms = sch.dendrogram(sch.linkage(X,method = 'ward',))
plt.title('Dendograms')
plt.xlabel('Clusters')
plt.ylabel('Euclidean Distance')
plt.show()

# Fitting Hierarchial clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity = 'euclidean',linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,color='red',label='Carefull')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,color='blue',label='Standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,color='green',label='Target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,color='cyan',label='Careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,color='magenta',label='Sensible')
#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,color='yellow',label='Centroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()

