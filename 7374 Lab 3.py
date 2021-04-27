#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from numpy import genfromtxt
# Import data as dataframe
board_game = pd.read_csv("board_games.csv")


# In[6]:


# Data Check/Preview
board_game.head(5)


# In[7]:


# Dataset information
board_game.describe()


# In[8]:


# Remove rows with missing values
board_game = board_game.dropna()
# Drop rows where the users_rated is 0
new_board_game = board_game[board_game["users_rated"] !=0 ]


# In[9]:


# Histogram of average_rating
data = np.array(new_board_game['average_rating'])
fig, ax = plt.subplots(figsize =(10, 7)) 
ax.hist(data, bins = 5) 

plt.xlabel('ratings(1-10)')
plt.ylabel('Nuumber of ratings')
plt.title('Average Rating')
plt.show() 


# In[10]:


# standard deviation
print("Standard Deviation of average rating is % s " 
                % (statistics.stdev(data))) 
# mean
print("Mean of average rating is % s " 
                % (statistics.mean(data)))


# In[11]:


# Correlation matrix/Heat map for visualization

result = new_board_game.corr()
#print(result)
sns.heatmap(new_board_game.corr())


# In[12]:


# Remove unnecessary columns and finalize our input dataset.
final_board_game = new_board_game.drop(new_board_game.columns[[0,1,3,4,5,6,7]], axis=1)


# In[13]:


# Split the data into 70% train and 30$ test.
X = final_board_game.drop(final_board_game.columns[[3]], axis=1)
Y = final_board_game[['average_rating']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Linear Regression Model
reg = linear_model.LinearRegression()
model = reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Print the accuracy
print ('The accuracy of the model is: ', model.score(X_test, y_test))

# MSE
result = sqrt(mean_squared_error(y_test, y_pred))
print('The RMSE value is: ', result)


# In[14]:


# K-fold cross validation
kf = KFold(n_splits=5)
scores = np.sqrt( -cross_val_score(reg, X, Y, scoring='neg_mean_squared_error', cv=kf))
print('The RMSE value is:',scores)


# In[15]:


# Ordinary least squares method
ols = sm.OLS(Y, X)
results = ols.fit()
print(results.summary())


# In[16]:


# normalize data
X_ntrain = preprocessing.scale(X_train)
y_ntrain = preprocessing.scale(y_train)
X_ntest = preprocessing.scale(X_test)
y_ntest = preprocessing.scale(y_test)


# In[17]:


# gradient descent method
sgd = linear_model.SGDRegressor()
sgdmodel = sgd.fit(X_ntrain, y_ntrain.reshape(-1,))
y_sgdpred = sgd.predict(X_ntest)

# Print the accuracy
print ('The accuracy of the model is: ', sgdmodel.score(X_ntest, y_ntest))

# RMSE
sgdresult = sqrt(mean_squared_error(y_ntest, y_sgdpred))
print('The RMSE value is: ', sgdresult)


# In[47]:


alter_final_game = final_board_game[['yearpublished', 'minage','users_rated','total_owners','total_traders','total_wanters','total_wishers','total_comments','total_weights','average_weight','average_rating']]
dataSet = alter_final_game.to_numpy()

alpha = 0.01
tol_L = 0.01
loss_new = tol_L+1

#normalization function
def normalization(X):
    means = X.mean(axis =0)
    stds = X.std(axis= 0)
    X= X - means[np.newaxis,:]
    X= X / stds[np.newaxis,:]
    return np.nan_to_num(X)

# Obtain the variable matrix from the read data
def getData(dataSet):
    m, n = np.shape(dataSet)
    trainData = np.ones((m, n))
    trainData[:,1:] = dataSet[:,:-1]
    trainLabel = dataSet[:,-1]
    return trainData, trainLabel

# Normalized the train data only
trainData, trainLabel = getData(dataSet)
#train_normalized = preprocessing.normalize(trainData)
train_normalized = normalization(trainData)
m, n = np.shape(train_normalized)
weight = np.ones(n)
grad = np.ones(n)

# Gradient algorithm function
def compute_grad(weight, x, y):
    grad = np.zeros(n)
    weight_t=weight.transpose()
    x_t = x.transpose()
    y_t = y.transpose()
    # grad[0] = 2. * np.mean(weight[0]+np.dot(weight,x_t) - y_t)
    grad = 2. * (np.dot(x_t ,(np.dot(x,weight) - y)))/m
    return np.array(grad)

def batchGradientDescent(x, y, weight, alpha, m, maxIterations):
    xTrains = x.transpose()
    # for i in range(0, maxIterations):
    hypothesis = np.dot(train_normalized, weight)
    loss = hypothesis - trainLabel
        # print loss
    gradient = np.dot(train_normalized.T, np.dot(train_normalized, weight)-trainLabel) / m
    weight = weight - alpha * gradient
    return weight

# update weight
def update_weight(weight, alpha, grad):
    new_weight = np.array(weight) - alpha * grad
    return new_weight

# RMSE
def rmse(weight, x, y):
    weight_t = weight.transpose()
    squared_err = (np.dot(x,weight_t)-y)**2
    res = np.sqrt(np.mean(squared_err))
    return res


i=1
while i < 50000:
    weight = update_weight(weight, alpha, grad)
    grad = compute_grad(weight, train_normalized, trainLabel)
    loss = loss_new
    loss_new = rmse(weight, train_normalized, trainLabel)
    if np.abs(loss_new - loss) < tol_L:
      i += 1
print(loss_new)


# In[22]:


# LassoCV
las = linear_model.Lasso()
lasmodel = las.fit(X_train, y_train)
y_laspred = las.predict(X_test)

# Print the accuracy
print ('The accuracy of the model is: ', lasmodel.score(X_test, y_test))

# RMSE
lasresult = sqrt(mean_squared_error(y_test, y_laspred))
print('The RMSE value is: ', lasresult)


# In[15]:


# RidgeCV()
rid = linear_model.RidgeCV()
ridmodel = rid.fit(X_train, y_train)
y_ridpred = rid.predict(X_test)

# Print the accuracy
print ('The accuracy of the model is: ', ridmodel.score(X_test, y_test))

# RMSE
ridresult = sqrt(mean_squared_error(y_test, y_ridpred))
print('The RMSE value is: ', ridresult)


# In[23]:


# Random Forest Regressor
rfr = RandomForestRegressor()
rfrmodel = rfr.fit(X_ntrain, y_ntrain.reshape(-1,))
y_rfrpred = rfr.predict(X_ntest)

# Print the accuracy
print ('The accuracy of the model is: ', rfrmodel.score(X_ntest, y_ntest))

# RMSE
rfrresult = sqrt(mean_squared_error(y_ntest, y_rfrpred))
print('The RMSE value is: ', rfrresult)


# In[24]:


#  Neural Networks MLPRegressor
mlp = MLPRegressor()
mlpmodel = mlp.fit(X_ntrain, y_ntrain.reshape(-1,))
y_mlppred = mlp.predict(X_ntest)

# Print the accuracy
print ('The accuracy of the model is: ', mlpmodel.score(X_ntest, y_ntest))

# RMSE
mlpresult = sqrt(mean_squared_error(y_ntest, y_mlppred))
print('The RMSE value is: ', mlpresult)


# In[ ]:




