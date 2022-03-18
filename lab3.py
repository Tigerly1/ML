#!/usr/bin/env python
# coding: utf-8

# In[257]:


#!/usr/bin/env python
# coding: utf-8


# In[7]:

# In[258]:


import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y}) 
df.to_csv('dane_do_regresji.csv',index=None)
df.plot.scatter(x='x',y='y')


# In[259]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['x']], df[['y']], test_size=0.2, shuffle=False)


# In[260]:


df_mse_score = pd.DataFrame(None, columns = ["train_mse", "test_mse"])
df_mse_score


# In[261]:


regressors = []


# # Linear Regression

# In[73]:

# In[262]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# In[74]:

# In[263]:


lin_reg.predict(X_test).reshape(1,-1)[0], X_test.values.reshape(1,-1)[0]


# In[93]:

# In[264]:


lin_predict_test = lin_reg.predict(X_test)
lin_predict_train = lin_reg.predict(X_train)
df2 = pd.DataFrame({'x':X_test.values.reshape(1,-1)[0], 'y':lin_predict_test.reshape(1,-1)[0]})
df2.plot.scatter(x='x',y='y')


# In[95]:

# In[265]:


from sklearn.metrics import mean_squared_error
test_mse_lin = mean_squared_error(y_test, lin_predict_test)
train_mse_lin = mean_squared_error(y_train, lin_predict_train)


# In[266]:


test_mse_lin, train_mse_lin


# In[267]:


df_mse_score.loc["lin_reg", "train_mse"] = train_mse_lin
df_mse_score.loc["lin_reg", "test_mse"] = test_mse_lin
df_mse_score


# In[268]:


regressors.append((lin_reg, None))


# # KNN 3

# In[98]:

# In[269]:


import sklearn.neighbors
knn_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train)
knn_predict_test = knn_reg.predict(X_test)
knn_predict_train = knn_reg.predict(X_train)


# In[90]:

# In[270]:


df3 = pd.DataFrame({'x':X_test.values.reshape(1,-1)[0], 'y':knn_predict_test.reshape(1,-1)[0]})
df3.plot.scatter(x='x',y='y')


# In[271]:


from matplotlib import pyplot

pyplot.plot(X_train, knn_predict_train, ".")
pyplot.plot(X_train, y_train, ".")


# In[101]:

# In[272]:


test_mse_knn3 = mean_squared_error(y_test, knn_predict_test)
train_mse_knn3 = mean_squared_error(y_train, knn_predict_train)


# In[273]:


test_mse_knn3, train_mse_knn3


# In[274]:


df_mse_score.loc["knn_3_reg", "train_mse"] = train_mse_knn3
df_mse_score.loc["knn_3_reg", "test_mse"] = test_mse_knn3
df_mse_score


# In[275]:


regressors.append((knn_reg, None))


# ### KNN 5

# In[276]:


import sklearn.neighbors
knn_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)
knn_predict_test = knn_reg.predict(X_test)
knn_predict_train = knn_reg.predict(X_train)


# In[107]:

# In[277]:


df4 = pd.DataFrame({'x':X_test.values.reshape(1,-1)[0], 'y':knn_predict_test.reshape(1,-1)[0]})
df4.plot.scatter(x='x',y='y')


# In[108]:

# In[278]:


test_mse_knn5 = mean_squared_error(y_test, knn_predict_test)
train_mse_knn5 = mean_squared_error(y_train, knn_predict_train)


# In[279]:


test_mse_knn5, train_mse_knn5


# In[280]:


df_mse_score.loc["knn_5_reg", "train_mse"] = train_mse_knn5
df_mse_score.loc["knn_5_reg", "test_mse"] = test_mse_knn5
df_mse_score


# In[281]:


regressors.append((knn_reg, None))


# In[109]:

# # Polynomial

# In[126]:

# In[282]:


from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)
lin_r_predict_test= lin_reg.predict(poly_features.fit_transform(X_test))
lin_r_predict_train = lin_reg.predict(poly_features.fit_transform(X_train))


# In[117]:

# In[283]:


df5 = pd.DataFrame({'x':X_test.values.reshape(1,-1)[0], 'y':lin_r_predict_test.reshape(1,-1)[0]})
df5.plot.scatter(x='x',y='y')


# In[118]:

# In[284]:


test_mse_poly2 = mean_squared_error(y_test, lin_r_predict_test)
train_mse_poly2 = mean_squared_error(y_train, lin_r_predict_train)
test_mse_poly2, train_mse_poly2


# In[285]:


df_mse_score.loc["poly_2_reg", "train_mse"] = train_mse_poly2
df_mse_score.loc["poly_2_reg", "test_mse"] = test_mse_poly2
df_mse_score


# In[286]:


regressors.append((lin_reg, poly_features))


# In[287]:


from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X_train)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)
lin_r_predict_test= lin_reg.predict(poly_features.fit_transform(X_test))
lin_r_predict_train = lin_reg.predict(poly_features.fit_transform(X_train))


# In[288]:


df5 = pd.DataFrame({'x':X_test.values.reshape(1,-1)[0], 'y':lin_r_predict_test.reshape(1,-1)[0]})
df5.plot.scatter(x='x',y='y')


# In[289]:


test_mse_poly2 = mean_squared_error(y_test, lin_r_predict_test)
train_mse_poly2 = mean_squared_error(y_train, lin_r_predict_train)
test_mse_poly2, train_mse_poly2


# In[290]:


df_mse_score.loc["poly_3_reg", "train_mse"] = train_mse_poly2
df_mse_score.loc["poly_3_reg", "test_mse"] = test_mse_poly2
df_mse_score


# In[291]:


regressors.append((lin_reg, poly_features))


# In[292]:


from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=4 , include_bias=False)
X_poly = poly_features.fit_transform(X_train)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)
lin_r_predict_test= lin_reg.predict(poly_features.fit_transform(X_test))
lin_r_predict_train = lin_reg.predict(poly_features.fit_transform(X_train))


# In[293]:


df5 = pd.DataFrame({'x':X_test.values.reshape(1,-1)[0], 'y':lin_r_predict_test.reshape(1,-1)[0]})
df5.plot.scatter(x='x',y='y')


# In[294]:


test_mse_poly2 = mean_squared_error(y_test, lin_r_predict_test)
train_mse_poly2 = mean_squared_error(y_train, lin_r_predict_train)
test_mse_poly2, train_mse_poly2


# In[295]:


df_mse_score.loc["poly_4_reg", "train_mse"] = train_mse_poly2
df_mse_score.loc["poly_4_reg", "test_mse"] = test_mse_poly2
df_mse_score


# In[296]:


regressors.append((lin_reg, poly_features))


# In[297]:


from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=5, include_bias=False)
X_poly = poly_features.fit_transform(X_train)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)
lin_r_predict_test= lin_reg.predict(poly_features.fit_transform(X_test))
lin_r_predict_train = lin_reg.predict(poly_features.fit_transform(X_train))


# In[297]:





# In[298]:


df5 = pd.DataFrame({'x':X_test.values.reshape(1,-1)[0], 'y':lin_r_predict_test.reshape(1,-1)[0]})
df5.plot.scatter(x='x',y='y')


# In[299]:


test_mse_poly2 = mean_squared_error(y_test, lin_r_predict_test)
train_mse_poly2 = mean_squared_error(y_train, lin_r_predict_train)
test_mse_poly2, train_mse_poly2


# In[300]:


df_mse_score.loc["poly_5_reg", "train_mse"] = train_mse_poly2
df_mse_score.loc["poly_5_reg", "test_mse"] = test_mse_poly2
df_mse_score


# In[301]:


regressors.append((lin_reg, poly_features))


# In[302]:


import pickle
pickle.dump(df_mse_score, open("mse.pkl", "wb"))


# In[303]:


regressors


# In[304]:


import pickle
pickle.dump(regressors, open("reg.pkl", "wb"))

