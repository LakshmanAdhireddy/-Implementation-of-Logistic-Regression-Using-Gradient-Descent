# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages required.

2.Read the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Lakshman
RegisterNumber:  212222240001
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]
print("Array of X") 
X[:5]
print("Array of y") 
y[:5]
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
print("Exam 1- score Graph")
plt.show()
def sigmoid(z):
    return 1/(1+np.exp(-z))
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
print("Sigmoid function graph")
plt.show()
def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print("X_train_grad value")
print(J)
print(grad)
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print("Y_train_grad value")
print(J)
print(grad)
def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad 
   
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(" Print res.x")
print(res.fun)
print(res.x)   
def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")![image](https://github.com/LakshmanAdhireddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707265/a8dc653e-7941-4bed-8562-2b2cd9593b0f)

    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()  
print("Decision boundary - graph for exam score")
plotDecisionBoundary(res.x,X,y)
prob=sigmoid(np.dot(np.array([1, 45, 85]),res.x))
print("Proability value ")
print(prob)
def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
print("Prediction value of mean")
np.mean(predict(res.x,X)==y)
```
## Output:
### Array of X:
![image](https://github.com/LakshmanAdhireddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707265/1fffbbdc-9ce6-450e-8f65-81f7575a3000)

### Array of Y:
![image](https://github.com/LakshmanAdhireddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707265/b1e4fa82-626b-4bab-a35d-436e559d522f)

### Score Graph:
![image](https://github.com/LakshmanAdhireddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707265/8a5ee75c-1091-4926-8a45-bd9c5bd3505e)

### Sigmoid Function Graph:
![image](https://github.com/LakshmanAdhireddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707265/c5bb062d-a9b8-486c-a561-9177f270a707)

### X_train_grad Value:
![image](https://github.com/LakshmanAdhireddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707265/b538a87e-9210-40ea-be65-d6fbdb016ae0)

### Y_train_grad Value:
![image](https://github.com/LakshmanAdhireddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707265/4cc8c9fa-79f1-4373-b7dc-c7962a1773cb)

### Print res_X:
![image](https://github.com/LakshmanAdhireddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707265/c4f37bf3-4886-4548-87ad-8622e79a69dd)

### Decision boundary:
![image](https://github.com/LakshmanAdhireddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707265/6e12c807-1878-46d5-a3e0-e6c2ec45e58c)

### Probability Value:
![image](https://github.com/LakshmanAdhireddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707265/70767cf2-cc32-46ba-815e-1b0044c95387)

### Prediction Value of Mean:
![image](https://github.com/LakshmanAdhireddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707265/869498e4-ef82-4fda-9c8a-396da1754d6f)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

