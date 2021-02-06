import numpy as np
import pandas as pd
from logistic_function import *

data = pd.read_csv('C:\\Users\\User\\Desktop\\projects\\breast-cancer-wisconsin.csv')
#additional way
#x=data.iloc[:375,1:10]
x1=data.iloc[:375,1]
x2=data.iloc[:375,2]
x3=data.iloc[:375,3]
x4=data.iloc[:375,4]
x5=data.iloc[:375,5]
x6=data.iloc[:375,6]
x7=data.iloc[:375,7]
x8=data.iloc[:375,8]
x9=data.iloc[:375,9]
x1=x1[:, np.newaxis]
x2=x2[:, np.newaxis]
x3=x3[:, np.newaxis]
x4=x4[:, np.newaxis]
x5=x5[:, np.newaxis]
x6=x6[:, np.newaxis]
x7=x7[:, np.newaxis]
x8=x8[:, np.newaxis]
x9=x9[:, np.newaxis]
features=[x1,x2,x3,x4,x5,x6,x7,x8,x9]
x=np.hstack(features)
x = x.transpose()

y=data.iloc[:375,10]
y[y==2]=0
y[y==4]=1
m = len(y)
y = y[:,np.newaxis]


omega = np.zeros([len(features),1])
b = 0.1
iterations = 1000
alpha = 0.1

J,omega,b=logistic_train(x,y,omega,b,iterations,alpha)
print( 'cost : ',J,'omega : ',omega, 'b: ' ,b)


print('::::::::::::::::::::::    TEST     :::::::::::::::::::::::')

x=data.iloc[375:,1:10]
y=data.iloc[375:,10]
y[y==2]=0
y[y==4]=1
length = len(y)
y = y[:,np.newaxis]
y=y.transpose()
x = x.transpose()

print(logistic_testing(x,y,b,omega, length))


