#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1)


# In[8]:


print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))


# In[27]:


import pandas as pd
y = pd.Series(mnist.target.astype(np.uint8))
X = mnist.data
y = y.sort_values(ascending=True)


# In[28]:


y.index


# In[30]:


X.reindex(y.index)


# In[30]:


X.reindex(y.index)


# In[31]:


X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[33]:


y_train, y_test


# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[41]:


pd.unique(y_train.sort_values()), pd.unique(y_test.sort_values())


# In[45]:


from sklearn.linear_model import SGDClassifier
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)


# In[46]:


a = sgd_clf.predict(X_train)
b = sgd_clf.predict(X_test)


# In[72]:


pickle.dump([sgd_clf.score(X_train, y_train_0), sgd_clf.score(X_test, y_test_0)], open("sgd_acc.pkl", "wb"))


# In[75]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


# In[79]:


y_train_score = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, n_jobs=-1)


# In[80]:


pickle.dump(y_train_score, open("sgd_cva.pkl", "wb"))


# In[81]:


sgd_clf_all = SGDClassifier(random_state=42)
sgd_clf_all.fit(X_train, y_train)


# In[ ]:


y_train_predict_all = cross_val_predict(sgd_clf_all, X_train, y_train, cv=3, n_jobs=-1)


# In[ ]:


conf_mx = confusion_matrix(y_train, y_train_predict_all)


# In[ ]:


pickle.dump(conf_mx, open("sgd_cmx.pkl", "wb"))

