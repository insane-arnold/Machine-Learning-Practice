#importing libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


#imoprt dataset

dataset = pd.read_excel("kuiper.xlsx")
X = dataset.iloc[:,1:].values
Y = dataset.iloc[:,0].values


#Handling categorical variable 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cl1 = ColumnTransformer(transformers = [("encoder",OneHotEncoder(),[1])],remainder = 'passthrough')
X = cl1.fit_transform(X)

cl2 = ColumnTransformer(transformers = [("encoder",OneHotEncoder(),[7])],remainder = 'passthrough')
X = cl2.fit_transform(X)

cl3 = ColumnTransformer(transformers = [("encoder",OneHotEncoder(),[39])],remainder = 'passthrough')
X = cl3.fit_transform(X)

cl4 = ColumnTransformer(transformers = [("encoder",OneHotEncoder(),[86])],remainder = 'passthrough')
X = cl4.fit_transform(X)


#Splitting the dataset

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)


#Creating a model 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


#Prediction 

Y_pred = regressor.predict(X_test)

#comparing the prediction with test set

comp = np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),axis = 1)

#getting the accuracy

from sklearn.metrics import r2_score
print(r2_score(Y_test,Y_pred))

#accuraccy_meacsure = 0.990763650748037