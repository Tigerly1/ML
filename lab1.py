#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os

if os.path.isdir('data'):
   os.mkdir('data')
   os.chdir('data')


# In[2]:


import urllib.request
urllib.request.urlretrieve('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz', 'housing.tgz')
    


# In[3]:


import tarfile
import gzip
import shutil
tar = tarfile.open('housing.tgz')
tar.extractall()

with open('housing.csv', 'rb') as f_in:
    with gzip.open('housing.csv.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# In[4]:


import pandas as pd
df = pd.read_csv('housing.csv.gz')


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.ocean_proximity.value_counts()


# In[8]:


df.ocean_proximity.describe()


# In[9]:


import matplotlib.pyplot as plt 
df.hist(bins=50, figsize=(20,15))
plt.savefig('obraz1.png')


# In[10]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.1, figsize=(7,4))
plt.savefig('obraz2.png')


# In[11]:


# potrzebne ze względu na argument cmap
df.plot(kind="scatter", x="longitude", y="latitude",
     alpha=0.4, figsize=(7,3), colorbar=True,
     s=df["population"]/100, label="population", 
     c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('obraz3.png')


# In[12]:


df.corr()["median_house_value"].sort_values(ascending=False)


# In[13]:


df.corr()["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={"atrybut":"index", "wspolczynnik_korelacji":"median_house_value"}).to_csv('korelacja.csv', index=False)


# In[15]:


import seaborn as sns
sns.pairplot(df)


# In[16]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df,
                                       test_size=0.2,
                                       random_state=42)
len(train_set),len(test_set)


# In[18]:


train_set.corr()


# In[19]:


test_set.corr()


# In[20]:


import pickle

pickle.dump(train_set, open("train_set", "wb"))
pickle.dump(test_set, open("test_set", "wb"))


# In[ ]:




