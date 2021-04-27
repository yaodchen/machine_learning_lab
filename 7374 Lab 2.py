#!/usr/bin/env python
# coding: utf-8

# In[403]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from random import randrange
from sklearn.linear_model import LinearRegression

# Import data
df = pd.read_csv("lab2.csv")


# In[404]:


# Data with eight predictors and two predictors
eight_model = df
two_model = df[['y','x1','x2']]


# In[405]:


# Defind LinearRegression model to make prediction
def linearSSE(x_train,y_train,x_test,y_test):
    # Let x be the input variable
    # Let y be label
    model=LinearRegression().fit(x_train,y_train)
    r_sq=model.score(x_train,y_train)
    y_pred=model.predict(x_test)
    SSE=np.sqrt(np.sum(np.square(y_test - y_pred)))
    return SSE


# In[406]:


# Define Cross Validation Split Function
def cross_validation_split(dataset, folds):
        dataset_split = []
        df_copy = dataset
        fold_size = int(df_copy.shape[0] / folds)
        
        # save each fold
        for i in range(folds):
            fold = []
            while len(fold) < fold_size:
                r = randrange(df_copy.shape[0])
                index = df_copy.index[r]
                fold.append(df_copy.loc[index].values.tolist())
                df_copy = df_copy.drop(index) 
            dataset_split.append(np.asarray(fold))
            
        return dataset_split


# In[407]:


# K Fold Cross Validation Function
def kfoldCrossValidation(dataset, f=10, k=10):
    data=cross_validation_split(dataset,f)
    # Set training and test data
    for i in range(f):
        lst = list(range(f))
        lst.pop(i)
        for j in lst :
            if j == lst[0]:
                crossVal = data[j]
            else:    
                crossVal=np.concatenate((crossVal,data[j]), axis=0)
        # Apply the eight predictors model:
            result1 = linearSSE(crossVal[:,1:],crossVal[:,0],data[i][:,1:],data[i][:,0])
        # Apply the two predictors model:
            result2 = linearSSE(crossVal[:,1:3],crossVal[:,0],data[i][:,1:3],data[i][:,0])
    print("SSE for Eight_Model is: " + str(result1))
    print("SSE for Two_Model is: " + str(result2))


# In[408]:


# Replicate the cross_validation 20 times and print the SSE score for two models.
count = 1
while count <= 20:
    print("Iteration "+str(count))
    print(kfoldCrossValidation(df, f=10, k=10))
    count = count + 1


# In[ ]:




