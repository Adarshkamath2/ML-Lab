#!/usr/bin/env python
# coding: utf-8

# # KNN with Glass Dataset

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("./glass.csv")
df.head()


# In[3]:


#Euclidean Distance

def ec(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


# In[4]:


from collections import Counter

class KNN:
    def __init__(self,k=3):
        self.k=k

    def fit(self,X,y):
        self.X_train=X
        self.y_train=y 

    def predict(self,X):
        predictions=[self._predict(x) for x in X]
        return predictions

    def _predict(self,x):
        #Compute distance from one given point to all the points in X_train
        distances=[ec(x1=x,x2=x_train) for x_train in self.X_train]

        #Get k closest indices and labels
        k_indices=np.argsort(distances)[:self.k]
        k_labels=[self.y_train[i] for i in k_indices]

        #Get most common class label
        co=Counter(k_labels).most_common()
        return co[0][0]


# In[5]:


#Split Data

X=df.drop("Type",axis=1).values
y=df['Type'].values
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=40)


# In[6]:


#Fit Model

clf=KNN(k=3)
clf.fit(X_train,Y_train)
predictions=clf.predict(X_test)
print(predictions)
plt.scatter(X[:,2],X[:,3],c=y)
plt.show()


# In[7]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred=predictions,y_true=Y_test))

