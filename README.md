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
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SANTHOSH D
RegisterNumber: 212223220099
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

DATASET
![image](https://github.com/user-attachments/assets/fc4137c7-6cba-452b-9d27-aba232586129)

HEAD VALUES

![image](https://github.com/user-attachments/assets/ef42f8a5-799f-40f8-a894-a115d47f31a5)

TRAIL VALUES

![image](https://github.com/user-attachments/assets/f37a6c65-a731-47b0-ae11-e1ade28fa835)

X AND Y VALUES

![image](https://github.com/user-attachments/assets/a7ff385c-de92-4df5-9a91-735896a14b1f)

PREDICATION VALUES OF X AND Y

![image](https://github.com/user-attachments/assets/334cfcbb-00d9-41d4-9844-f37b9d43a2bb)

MSE,MAE and RMSE

![image](https://github.com/user-attachments/assets/716d9e8b-932c-42d8-8b4e-ebf2be0f1f54)

TRAINING SET

![image](https://github.com/user-attachments/assets/e0afd353-6a0b-4312-abe4-872749e81607)

TESTING SET

![image](https://github.com/user-attachments/assets/a040063e-376e-4412-a1eb-ce487d8b900d)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
