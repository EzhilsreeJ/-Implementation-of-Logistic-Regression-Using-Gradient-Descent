# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Load the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: EZHIL SREE J
RegisterNumber:  212223230056
*/
```
```
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```
```
df=pd.read_csv("Placement_Data_Full_Class .csv")
df.head()
```
![image](https://github.com/user-attachments/assets/68f26053-14a6-4a3c-abc4-bb890301ec2a)
```
df.tail()
```
![image](https://github.com/user-attachments/assets/76d14937-5ddd-4dd4-8287-df89180a88d6)
```
df.info()
```
![image](https://github.com/user-attachments/assets/21434dae-a403-4ee4-8d69-492ba6b2ab62)
```
df.drop('sl_no',axis=1)
```
![image](https://github.com/user-attachments/assets/36dd9395-7961-4e91-86f4-7b1ac98c79f4)
```
df.drop('sl_no',axis=1,inplace=True)
```
```
df["gender"]=df["gender"].astype('category')
df["ssc_b"]=df["ssc_b"].astype('category')
df["hsc_b"]=df["hsc_b"].astype('category')
df["degree_t"]=df["degree_t"].astype('category')
df["workex"]=df["workex"].astype('category')
df["specialisation"]=df["specialisation"].astype('category')
df["status"]=df["status"].astype('category')
df["hsc_s"]=df["hsc_s"].astype('category')
df.dtypes
```
![image](https://github.com/user-attachments/assets/f8cdf7a0-6b3c-4c3c-a890-932d593c4d36)
```
df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes 
```
```
df.info()
```
![image](https://github.com/user-attachments/assets/bad44f3c-7f70-4b82-95e5-294edf81e7ac)
```
df.head()
```
![image](https://github.com/user-attachments/assets/f31c2530-9bfd-4873-ac9f-864d186f99e4)

```
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
X
Y
```
![image](https://github.com/user-attachments/assets/60b78cd7-a652-4a1c-92fe-194d4569e6f9)

```
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid (z):
    return 1/(1+np.exp(-z))
def loss (theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha, num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h= sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y)/m
        theta -= alpha * gradient 
    return theta
theta= gradient_descent(theta,X,y,alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h= sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:",accuracy)
```
![image](https://github.com/user-attachments/assets/c1bc4404-29e2-4dc1-83d8-e119d878ee8f)
```
print(y_pred)
```
![image](https://github.com/user-attachments/assets/c2a25c3d-43dd-4ce5-b8d9-310ceee2817b)
```
print(Y)
```
![image](https://github.com/user-attachments/assets/40a2be56-fc52-4db8-994b-be23fee2c994)
```
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/d849c110-cf4a-4733-a1a3-7a635258a763)
```
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/5835b3dc-f6c2-4a10-8b35-2a542b056899)


## Output:


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

