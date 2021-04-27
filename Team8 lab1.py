import pandas as pd
import pylab as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from numpy import genfromtxt

# read the csv file
df = pd.read_csv("C:/Users/18572/Desktop/lab1/housing.csv")
dataPath = "C:/Users/18572/Desktop/lab1/housing.csv"
dataSet = genfromtxt(dataPath, delimiter=",", skip_header=1)


alpha = 0.0004
tol_L = 0.005
loss_new = tol_L+1
#other learning rates and tolerance for housing
#alpha = 0.05, tol_L = 0.1
#alpha = 0.01, tol_L = 0.1
#learning rates and tolerance for Concrete
#alpha = 0.0007，tol_L = 0.0001
#alpha = 0.05，tol_L = 0.01
#alpha = 0.1，tol_L = 0.01
#learning rates and tolerance for Yacht
#alpha = 0.001，tol_L = 0.001
#alpha = 0.01，tol_L = 0.05
#alpha = 0.1，tol_L = 0.05



# apply the z-score normalization method
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std

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
    print('Round %s Diff RMSE %s'%(i, abs(loss_new - loss)))
print(weight)
print(loss_new)


#normal equation
trainData_transpose=trainData.transpose()
trainData_transpose_trainData=np.dot(trainData_transpose,trainData)
trainData_transpose_trainLabel=np.dot(trainData_transpose,trainLabel)
trainData_transpose_trainData_inverse=np.linalg.inv(trainData_transpose_trainData)
weight2=np.dot(trainData_transpose_trainData_inverse, trainData_transpose_trainLabel)
print(weight2)





