#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn import ensemble
# from sklearn.metrics import mean_absolute_error
# import joblib as joblib


# In[2]:


df = pd.read_csv('G:/Data Engineering/Python/Machine Learning/ML Book/Dataset/Melbourne_housing_FULL.csv')


# In[3]:


print(df.head(n=5))


# In[4]:


# del df['Address']
# del df['Method']
# del df['SellerG']
# del df['Date']
# del df['Postcode']
# del df['Lattitude']
# del df['Longtitude']
# del df['Regionname']
# del df['Propertycount']
#
#
# # In[5]:
#
#
# print(df)
#
#
# # In[6]:
#
#
# df.dropna(axis = 0, how = 'any', thresh = None, subset = None, inplace = True)
#
#
# # In[7]:
#
#
# df
#
#
# # In[8]:
#
#
# features_df = pd.get_dummies(df,columns = ['Suburb','CouncilArea','Type'])
#
#
# # In[9]:
#
#
# features_df
#
#
# # In[10]:
#
#
# del features_df['Price']
#
#
# # In[11]:
#
#
# features_df
#
#
# # In[12]:
#
#
# X = pd.DataFrame(features_df)
# Y = pd.DataFrame(df['Price'])
#
#
# # In[13]:
#
#
# X
#
#
# # In[14]:
#
#
# Y
#
#
# # In[15]:
#
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
#
#
# # In[16]:
#
#
# model = ensemble.GradientBoostingRegressor(
# n_estimators = 150,
# learning_rate = 0.1,
# max_depth = 30,
# min_samples_split = 4,
# min_samples_leaf = 6,
# max_features = 0.6,
# loss = 'huber')
# model.fit(X_train,np.ravel(Y_train))
# joblib.dump(model,'house_trained_model.pkl')
#
#
# # In[17]:
#
#
# mse = mean_absolute_error(Y_train, model.predict(X_train))
# print("Training Set Mean Absolute Error: %.2f" % mse)
#
# mse = mean_absolute_error(Y_test, model.predict(X_test))
# print("Test Set Mean Absolute Error: %.2f" % mse)

