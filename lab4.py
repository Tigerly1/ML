#!/usr/bin/env python
# coding: utf-8

# In[16]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[17]:


data_iris = datasets.load_iris(as_frame=True)


# In[18]:


X,y = data_breast_cancer["data"][["mean area", "mean smoothness"]], data_breast_cancer["target"]


# In[19]:


print(X)


# In[20]:


import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[21]:


from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
lin_svc = LinearSVC(loss="hinge", max_iter=696969)
svm_clf = Pipeline([
        ("linear_svc", lin_svc)])
lin_svc = LinearSVC(loss="hinge", max_iter=696969)
svm_clf_scaler = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", lin_svc)])
svm_clf.fit(X_train,y_train)
svm_clf_scaler.fit(X_train,y_train)


# In[22]:


a = [svm_clf.score(X_train, y_train), svm_clf.score(X_test, y_test)]


# In[23]:


b = [svm_clf_scaler.score(X_train, y_train), svm_clf_scaler.score(X_test, y_test)]


# In[24]:


a += b
print(a)


# In[25]:


import pickle
pickle.dump(a, open("bc_acc.pkl", "wb"))


# In[26]:


X,y = data_iris["data"], (data_iris["target"]==2).astype(np.int32)


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
x_train_props = X_train[['petal length (cm)', 'petal width (cm)']]
x_train_props_test = X_test[['petal length (cm)', 'petal width (cm)']]


# In[28]:


lin_svc = LinearSVC(loss="hinge")
svm_clf_2 = Pipeline([("linear_svc", lin_svc)])
lin_svc = LinearSVC(loss="hinge")
svm_clf_scaler_2 = Pipeline([("scaler", StandardScaler()),("linear_svc", lin_svc)])
svm_clf_2.fit(x_train_props,y_train)
svm_clf_scaler_2.fit(x_train_props,y_train)


# In[29]:


result = [svm_clf_2.score(x_train_props, y_train), svm_clf_2.score(x_train_props_test, y_test), svm_clf_scaler_2.score(x_train_props, y_train), svm_clf_scaler_2.score(x_train_props_test, y_test)]
result


# In[30]:


with open("iris_acc.pkl", "wb") as fout:
        pickle.dump(result, fout)


# In[30]:





# In[30]:




