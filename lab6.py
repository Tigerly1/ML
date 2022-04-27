#!/usr/bin/env python
# coding: utf-8

# In[233]:


import pandas as pd
from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[234]:


data_breast_cancer.data


# In[235]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_breast_cancer.data.loc[:, ["mean texture", "mean symmetry"]], data_breast_cancer.target, test_size=0.2)
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[236]:


log_clf = LogisticRegression(solver="lbfgs",
                             random_state=42)
dctree_clf =  DecisionTreeClassifier()

neigh_clf = KNeighborsClassifier()
first = []
first_classificators = []
voting_clf_hard = VotingClassifier(
    estimators=[('lr', log_clf),
                ('rf', dctree_clf),
                ('knn', neigh_clf)],
    voting='hard')

voting_clf_soft = VotingClassifier(
    estimators=[('lr', log_clf),
                ('rf', dctree_clf),
                ('knn', neigh_clf)],
    voting='soft')
for clf in ( dctree_clf, log_clf,neigh_clf, voting_clf_hard, voting_clf_soft):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    first_classificators.append(clf)
    first.append((accuracy_score(y_train, y_pred_train),accuracy_score(y_test, y_pred)))
    


# In[237]:


pickle.dump(first, open("acc_vote.pkl", "wb"))
pickle.dump(first_classificators, open("vote.pkl", "wb"))


# In[238]:


acc_bag = []
from sklearn.ensemble import BaggingClassifier



bag_clf = BaggingClassifier(
DecisionTreeClassifier(), n_estimators=30,
max_samples=100, bootstrap=True)
bag_clf.fit(X_train, y_train)
acc_bag.append((bag_clf.score(X_train, y_train),bag_clf.score(X_test, y_test)))


# In[239]:


bag_clf_zero = BaggingClassifier(
DecisionTreeClassifier(), n_estimators=30,
max_samples=0.5, bootstrap=True)
bag_clf_zero.fit(X_train, y_train)
acc_bag.append((bag_clf_zero.score(X_train, y_train),bag_clf_zero.score(X_test, y_test)))


# In[240]:


bag_clf_first = BaggingClassifier(
DecisionTreeClassifier(), n_estimators=30,
max_samples=100, bootstrap=False)
bag_clf_first.fit(X_train, y_train)
acc_bag.append((bag_clf_first.score(X_train, y_train),bag_clf_first.score(X_test, y_test)))


# In[241]:


bag_clf_second = BaggingClassifier(
DecisionTreeClassifier(), n_estimators=30,
max_samples=0.5, bootstrap=False)
bag_clf_second.fit(X_train, y_train)
acc_bag.append((bag_clf_second.score(X_train, y_train),bag_clf_second.score(X_test, y_test)))


# In[242]:


from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500,
                                 max_leaf_nodes=16)
rnd_clf.fit(X_train, y_train)
acc_bag.append((rnd_clf.score(X_train, y_train),rnd_clf.score(X_test, y_test)))


# In[243]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, data_breast_cancer.data.loc[:, ["mean texture", "mean symmetry"]], data_breast_cancer.target, cv=5)
acc_bag.append((scores.mean(), scores.mean()))


# In[244]:


from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(max_depth=2
                                 ,n_estimators=3,learning_rate=1.0)
gbrt.fit(X_train, y_train)
acc_bag.append((gbrt.score(X_train, y_train),gbrt.score(X_test, y_test)))


# In[245]:


bag = [bag_clf, bag_clf_zero, bag_clf_first, bag_clf_second, rnd_clf, clf, gbrt]


# In[246]:


pickle.dump(acc_bag, open("acc_bag.pkl", "wb"))
pickle.dump(bag, open("bag.pkl", "wb"))


# In[247]:



bag_clf_fea = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    max_samples=100, bootstrap=True, max_features=2)

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(data_breast_cancer.data, data_breast_cancer.target, test_size=0.2)
bag_clf_fea.fit(X_train_1, y_train_1)
acc_fea_list = [bag_clf_fea.score(X_train_1, y_train_1), bag_clf_fea.score(X_test_1, y_test_1)]


# In[248]:


pickle.dump(acc_fea_list, open("acc_fea.pkl", "wb"))
pickle.dump([bag_clf_fea], open("fea.pkl", "wb"))


# In[249]:


acc_fea_rank_list_acc_train = []
acc_fea_rank_list_acc_test = []
acc_fea_rank_list_names = []
for x, clf in enumerate(bag_clf_fea.estimators_):
    a , b = data_breast_cancer.data.columns[bag_clf_fea.estimators_features_[x][0]], data_breast_cancer.data.columns[bag_clf_fea.estimators_features_[x][1]]
    location = lambda y:  y.loc[:,[a,b]]

    y_pred = clf.predict(location(X_test_1))
    y_pred_train = clf.predict(location(X_train_1))
    acc_fea_rank_list_acc_train.append(accuracy_score(y_train_1.values, y_pred_train))
    acc_fea_rank_list_acc_test.append(accuracy_score(y_test_1.values, y_pred))
    acc_fea_rank_list_names.append([a,b])


# In[250]:


data = {'Train accuracy':acc_fea_rank_list_acc_train, 'Test accuracy':acc_fea_rank_list_acc_test, "feature_names":acc_fea_rank_list_names}
df_acc = pd.DataFrame(data)
df_acc


# In[251]:


df_acc = df_acc.sort_values(by=['Train accuracy', 'Test accuracy'], ascending=False)
df_acc


# In[252]:



pickle.dump(df_acc, open("acc_fea_rank.pkl", "wb"))

