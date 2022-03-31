#!/usr/bin/env python
# coding: utf-8

# In[67]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True) 
print(data_breast_cancer['data'])


# In[67]:





# In[68]:


X, y = data_breast_cancer['data'], data_breast_cancer['target']


# In[68]:







# In[69]:



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
tree_clf = DecisionTreeClassifier(max_depth=4,
random_state=38)
tree_clf.fit(X_train, y_train)
pred_test = tree_clf.predict(X_test)
pred_train = tree_clf.predict(X_train)
from sklearn.metrics import f1_score
f1_train, f1_test = f1_score(pred_train, y_train), f1_score(pred_test, y_test)
print(f1_score(pred_test, y_test), f1_score(pred_train, y_train))
score_train, score_test = tree_clf.score(X_train, y_train), tree_clf.score(X_test, y_test)
print(tree_clf.score(X_train, y_train), tree_clf.score(X_test, y_test))


# In[70]:


import pickle
pickle.dump([4, f1_train, f1_test, score_train, score_test], open("f1acc_tree.pkl", "wb"))


# In[71]:


from sklearn.tree import export_graphviz
import graphviz
export_graphviz(
        tree_clf,
        out_file='bc',
        feature_names=data_breast_cancer['data'].columns,
        rounded=True,
        filled=True)
graph = graphviz.Source.from_file('bc')
graph
graph.render(format='png', directory='.')


# In[71]:





# In[72]:


###REGRESJA


# In[73]:


import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4 
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')


# In[73]:





# In[74]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[['x']], df[['y']], test_size=0.2, random_state=499)
tree_reg = DecisionTreeRegressor(max_depth=5, random_state=499)
tree_reg.fit(X_train, y_train)
pred_train = tree_reg.predict(X_train)
pred_test = tree_reg.predict(X_test)
from sklearn.metrics import mean_squared_error
mse_train = mean_squared_error(y_train, pred_train)
mse_test = mean_squared_error(y_test, pred_test)
print(mse_train, mse_test)


# In[75]:


from sklearn.tree import export_graphviz
import graphviz
export_graphviz(
        tree_reg,
        out_file='reg',
        rounded=True,
        filled=True)
graph = graphviz.Source.from_file('reg')
graph
graph.render(format='png', directory='.')


# In[76]:


import pickle
pickle.dump([5, mse_train, mse_test], open("mse_tree.pkl", "wb"))

