#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import math
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

fulldata=pd.read_csv('AdmissionDataset/data.csv')
fulldata=fulldata.drop(columns=['Serial No.'])

data,testdata=np.split(fulldata,[int(.8*len(fulldata))])

outputdata=data[data.columns[-1]]
inputdata = data[data.columns[0:7]]


# In[23]:


mean=np.mean(inputdata)
stdev=np.std(inputdata)


# In[24]:


mean


# In[25]:


col = inputdata['GRE Score']
normcol = [(1.0 * (c-mean['GRE Score']))/stdev['GRE Score'] for c in col]
print(normcol)


# In[26]:


ones=[]
for i in range(len(data)):
    ones.append(1)


# In[27]:


X=[]
X.append(ones)


# In[28]:


for i in range(len(inputdata.columns)):
    col = inputdata[inputdata.columns[i]]
    normcol = [(1.0 * (c-mean[i]))/stdev[i] for c in col]
    X.append(normcol)
X=np.array(X)


# In[29]:


B = np.ones(len(inputdata.columns)+1)
Y = np.array(outputdata)


# In[39]:


def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((np.dot(B,X) - Y) ** 2)
    J=J/m
    return J


# In[48]:


def MAE(X,Y,B):
    m = len(Y)
    J = np.sum((np.dot(B,X) - Y))
    J=J/m
    return J


# In[ ]:





# In[40]:


inital_cost = cost_function(X, Y, B)
print(inital_cost)


# In[41]:


def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(inputdata)
    
    for iteration in range(iterations):
        h = np.dot(B,X)
        loss = h - Y
        gradient = np.dot(loss,X.T) / m
        B = B - alpha * gradient
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B,cost_history


# In[42]:


B,cost_history = gradient_descent(X, Y, B, 0.01, 100)
print(B)


# In[43]:


def predict(row, B):
    y=B[0]
    i=1
    for r in row:
        y+=r*B[i]
        i+=1
    return y


# In[44]:


def validate(B,testdata):
    validatedata = []
    for i in range(len(testdata.columns)-1):
        col = testdata[testdata.columns[i]]
        normcol = [1.0 * (c-mean[i])/stdev[i] for c in col]
        validatedata.append(normcol)
    
    validatevector=np.array(validatedata).T
 
    actual=testdata[testdata.keys()[-1]]

    predicted=[]
    for i in range(len(validatevector)):
        row=validatevector[i]
        y=predict(row,B)
        predicted.append(y)
    
    print(r2_score(actual,predicted))


# In[45]:


validate(B,testdata)


# In[46]:


cost_history


# In[47]:


index =[]
for i in range(0,100):
    index.append(i)
plt.plot(index, cost_history)
plt.show()


# In[ ]:




