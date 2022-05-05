#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer()
from sklearn.datasets import load_iris 
data_iris = load_iris()


# In[57]:


import numpy as np


# In[3]:


X_breast = data_breast_cancer.data
y_breast = data_breast_cancer.target
X_iris = data_iris.data
y_iris = data_iris.target


# In[42]:


from sklearn.decomposition import PCA
pca = PCA(n_components=0.9)
X2D = pca.fit(X_iris)
print(pca.explained_variance_ratio_)


# In[43]:


pca = PCA(n_components=0.9)
X2D = pca.fit(X_breast)
print(pca.explained_variance_ratio_)


# In[53]:


pca.components_


# In[86]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
transformed_X_iris = scaler.fit_transform(X_iris)
pca_ir = PCA(n_components=0.9)
X2D_ir = pca_ir.fit_transform(transformed_X_iris)
print(pca_ir.explained_variance_ratio_)
pca_evr = pca_ir.explained_variance_ratio_


# In[ ]:


import pickle
pickle.dump(pca_evr, open("pca_ir.pkl", "wb"))


# In[87]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
transformed_X_breast = scaler.fit_transform(X_breast)
pca_br = PCA(n_components=0.9)
X2D_br = pca_br.fit_transform(transformed_X_breast)
print(pca_br.explained_variance_ratio_)
pca_evr = pca_br.explained_variance_ratio_


# In[ ]:


pickle.dump(pca_evr, open("pca_bc.pkl", "wb"))


# In[91]:


idx_bc = [ np.where(x==max(x))[0][0] for x in abs(pca_br.components_)]


# In[ ]:


pickle.dump(idx_bc, open("idx_bc.pkl", "wb"))


# In[92]:


idx_ir = [ np.where(x==max(x))[0][0] for x in abs(pca_ir.components_)]


# In[ ]:


pickle.dump(idx_ir, open("idx_ir.pkl", "wb"))


# In[ ]:





# In[71]:


import matplotlib.pyplot as plot
plot.scatter(transformed_X_breast[:,7], X2D_br[:,0]) 


# In[74]:


import matplotlib.pyplot as plot
plot.scatter(transformed_X_iris[:,2], X2D_ir[:,0]) 

