import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import math

data = pd.read_csv('E:\\Python\\Knn\\train.csv')

X = data.drop(['label'],axis = 1)
y = data['label']

from sklearn.model_selection import train_test_split as tts

train_x,test_x,train_y,test_y = tts(X,y,test_size = 0.3, random_state = 10)

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import metrics

def ecur(k):

    error_test = []

    for i in k:
        c = knn(n_neighbors=i)
        c.fit(train_x,train_y)
        tmp = c.predict(test_x)
        tmp = metrics.accuracy_score(tmp,test_y)
        error = 1-tmp
        error_test.append(error)
    return error_test

k = range(1,10)

test = ecur(k)

plt.plot(k, test)
plt.xlabel('K Neighbors')
plt.ylabel('Test error')
plt.title('Elbow curve for test')
plt.show()
m={}
for i in range(1,10):
    m[i]=np.interp(i,k,test)

val=1
for j in range(1,10):
    if(val>m[j]):
        val=m[j]
        num=j

# print(num,val)    
# test.sort()
# m=test[0]
# m=math.floor(m)
# print(m)
va = knn(n_neighbors=num)
va.fit(train_x,train_y)

pred = va.predict(test_x)

from sklearn.metrics import classification_report
print(classification_report(test_y, pred)) 