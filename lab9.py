#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


# In[2]:


iris = load_iris(as_frame=True)
pd.concat([iris.data, iris.target], axis=1).plot.scatter(
    x='petal length (cm)',
    y='petal width (cm)',
    c='target',
    colormap='viridis'
)


# In[3]:


df = pd.concat([iris.data, iris.target], axis=1)


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[["petal length (cm)","petal width (cm)"]], iris.target, test_size=0.2, random_state=69)


# In[5]:


y_train1 = y_train==0
y_train2 = y_train==1
y_train3 = y_train==2

y_test1 = y_test ==0
y_test2 = y_test ==1
y_test3 = y_test ==2


# In[6]:


from sklearn.linear_model import Perceptron

per_clf_1 = Perceptron()
per_clf_1.fit(X_train, y_train1)

per_clf_2 = Perceptron()
per_clf_2.fit(X_train, y_train2)

per_clf_3 = Perceptron()
per_clf_3.fit(X_train, y_train3)


# In[7]:


a_tr_1 = per_clf_1.score(X_train, y_train1)
a_te_1 = per_clf_1.score(X_test, y_test1)

a_tr_2 = per_clf_2.score(X_train, y_train2)
a_te_2 = per_clf_2.score(X_test, y_test2)

a_tr_3 = per_clf_3.score(X_train, y_train3)
a_te_3 = per_clf_3.score(X_test, y_test3)


# In[8]:


w_0 = per_clf_1.intercept_
w_1 = per_clf_2.intercept_
w_2 = per_clf_3.intercept_


# In[9]:


print(a_tr_1, a_te_1, a_tr_2, a_te_2, a_tr_3, a_te_3)


# In[10]:


per_acc = [(a_tr_1, a_te_1), (a_tr_2, a_te_2),(a_tr_3, a_te_3)]
per_acc


# In[11]:


per_wght = (w_0,w_1,w_2)


# In[12]:


print(per_wght)


# In[13]:


import pickle
pickle.dump(per_acc, open("per_acc.pkl", "wb"))


# In[14]:


import pickle
pickle.dump(per_wght, open("per_wght.pkl", "wb"))


# In[15]:


X = np.array([[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]])
y = np.array([0,
1, 1, 0])
per_clf_XOR = Perceptron()
per_clf_XOR.fit(X, y)
print(per_clf_XOR.coef_,per_clf_XOR.intercept_)


# In[16]:


import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential()
model.add(keras.layers.Dense(2, activation="tanh"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.2))
             

history = model.fit(X, y, epochs=100, verbose=False) 
print(history.history['loss'])

model.predict(X)


# In[17]:


weights = model.get_weights()


# In[18]:


import pickle
pickle.dump(weights, open("mlp_xor_weights.pkl.", "wb"))

