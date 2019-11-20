#!/usr/bin/env python
# coding: utf-8

# In[23]:


import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression 
#导入数据集iris  
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
dataset = pandas.read_csv(url)
dataset.columns = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width', 'class']
dataset['class'] = dataset['class'].apply(lambda x: x.split('-')[1])
print(dataset.describe())

#获取花卉花萼长度与花萼宽度数据集  
X = dataset['sepal_len']  
Y = dataset['sepal_width']
#三类鸢尾花的花萼长度与花萼宽度的散点图
get_ipython().run_line_magic('matplotlib', 'inline')
#plt.scatter(X, Y, c=iris.target, marker='x')
plt.scatter(X[:50], Y[:50], color='red', marker='o', label='setosa') #前50个样本
plt.scatter(X[50:100], Y[50:100], color='blue', marker='x', label='versicolor') #中间50个
plt.scatter(X[100:], Y[100:],color='green', marker='+', label='Virginica') #后50个样本
plt.legend(loc=2) #loc=1，2，3，4分别表示label在右上角，左上角，左下角，右下角
plt.show()

#获取花卉花瓣长度与花瓣宽度数据集    
M = dataset['petal_len']  
N = dataset['petal_width'] 
#三类鸢尾花的花瓣长度与花瓣宽度的散点图
get_ipython().run_line_magic('matplotlib', 'inline')
#plt.scatter(X, Y, c=iris.target, marker='x')
plt.scatter(M[:50], N[:50], color='red', marker='o', label='setosa') #前50个样本
plt.scatter(M[50:100], N[50:100], color='blue', marker='x', label='versicolor') #中间50个
plt.scatter(M[100:], N[100:],color='green', marker='+', label='Virginica') #后50个样本
plt.legend(loc=2) #loc=1，2，3，4分别表示label在右上角，左上角，左下角，右下角
plt.show()

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris   
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#载入数据集
iris = load_iris()
#划分训练与测试的数据集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target,test_size=0.3)

#逻辑回归模型
lr = LogisticRegression()
lr.fit(x_train, y_train)

#模型评估
Testgrades = lr.score(x_test, y_test)
Traingrades = lr.score(x_train, y_train)
print("TestScore: %.3f" %Testgrades)
print("TrainScore: %.3f" %Traingrades)
#正确率
y_predict = lr.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_predict)
print("accuracy: %.3f" %accuracy)

# TestScore: 0.867
# Traingrades: 0.962
# Accuracy: 0.867




