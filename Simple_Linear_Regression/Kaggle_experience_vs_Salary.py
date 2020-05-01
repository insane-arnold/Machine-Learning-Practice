#importing libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

#importing dataset 

dataset = pd.read_csv("c://Users//HP//Desktop//ML//ML practice//Simple_Linear_Regression//Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1:].values

#Splitting dataset into training and test set

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3)

#Building a Simple Linear Regression Model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#prediction

Y_pred = regressor.predict(X_test)

#Visualization

pt.scatter(X,Y,color = "red")
pt.plot(X,regressor.predict(X))  #plotting line of best fit
pt.show()

#calculating r2_score

from sklearn.metrics import r2_score
print(r2_score(Y_test,Y_pred))