# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Santhosh D
RegisterNumber:  212223220099

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
```

## Output:
![WhatsApp Image 2024-09-02 at 10 51 51_ff01643d](https://github.com/user-attachments/assets/1cb75479-d766-4a19-afb4-b4052f06ab39)
![WhatsApp Image 2024-09-02 at 10 52 54_19a869d4](https://github.com/user-attachments/assets/6b447b37-d74a-4a1a-80bb-8ee9ad4985d3)

![WhatsApp Image 2024-09-02 at 10 51 52_baadf86c](https://github.com/user-attachments/assets/db82b480-1015-49b4-8cd9-12c0f15d9f6e)

![WhatsApp Image 2024-09-02 at 10 51 47_1459f905](https://github.com/user-attachments/assets/5b82a600-d1d5-418b-9774-f57486112d2c)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
