#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
mnist = fetch_openml('mnist_784', version=1)


# ### Poprawne zbiory danych +97% acc

# In[37]:


from sklearn.model_selection import train_test_split
X, y = mnist.data, mnist.target.astype(np.uint8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[38]:


pd.unique(y_train.sort_values()), pd.unique(y_test.sort_values())


# In[40]:


from sklearn.linear_model import SGDClassifier
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)


# In[42]:


sgd_clf.score(X_train, y_train_0), sgd_clf.score(X_test, y_test_0)


# In[44]:


import pickle
pickle.dump([sgd_clf.score(X_train, y_train_0), sgd_clf.score(X_test, y_test_0)], open("sgd_acc.pkl", "wb"))


# In[50]:


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix


# In[47]:


y_train_score = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, n_jobs=-1)
y_train_score


# In[48]:


pickle.dump(y_train_score, open("sgd_cva.pkl", "wb"))


# In[52]:


sgd_clf_all = SGDClassifier(random_state=42)
sgd_clf_all.fit(X_train, y_train)


# In[53]:


y_train_predict_all = cross_val_predict(sgd_clf_all, X_train, y_train, cv=3, n_jobs=-1)
y_train_predict_all


# In[54]:


conf_mx = confusion_matrix(y_train, y_train_predict_all)
conf_mx


# In[ ]:


pickle.dump(conf_mx, open("sgd_cmx.pkl", "wb"))

