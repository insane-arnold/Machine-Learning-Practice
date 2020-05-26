# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
cl = ColumnTransformer(transformers = [("encoder",OneHotEncoder(),[3])],remainder = "passthrough")
X = cl.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 10)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

Y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
pred = np.concatenate((Y_pred.reshape(len(Y_pred),1),
                       Y_test.reshape(len(Y_test),1)),axis = 1)


#Measuring the r2_score
from sklearn.metrics import r2_score
print(r2_score(Y_test,Y_pred))