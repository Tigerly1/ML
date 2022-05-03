#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# In[3]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
clusters = []
sil_clusters = []
for x in range(8,13):
    kmeans = KMeans(n_clusters=x)
    kmeans.fit(X)
    clusters.append(kmeans)
    a = silhouette_score(X, kmeans.labels_)
    sil_clusters.append(a)



# In[ ]:


import pickle
pickle.dump(sil_clusters, open("kmeans_sil.pkl", "wb"))


# In[4]:


from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
km_10_confmat = set()
kmeans_10 = clusters[2]
y_pred = kmeans_10.predict(X)
conf = confusion_matrix(y, y_pred)
print(conf)
for row in conf:
    km_10_confmat.add(np.argmax(row))
    
print(km_10_confmat)
km_10_confmat = list(km_10_confmat)


# In[ ]:


pickle.dump(km_10_confmat, open("kmeans_argmax.pkl", "wb"))


# In[5]:


data = []
for x in X[:300]:
    for x2 in X:
        data.append(np.linalg.norm(x-x2))
    


# In[7]:


smallest = np.sort(data)[300:310]
smallest


# In[ ]:


pickle.dump(smallest, open("dist.pkl", "wb"))


# In[12]:


b = np.average(smallest[:3])
from sklearn.cluster import DBSCAN
s = b
dbscan_len = []
while s<=b+0.1*b:
    dbsc=DBSCAN(eps=s)
    dbsc.fit(X)
    dbscan_len.append(len(np.unique(dbsc.labels_)))
    s+=0.04*s


# In[ ]:


pickle.dump(dbscan_len, open("dbscan_len.pkl", "wb"))


# In[ ]:




