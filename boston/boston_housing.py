
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn.datasets as datasets

#机器算法模型
from sklearn.linear_model import LinearRegression


#切割训练数据和样本数据
from sklearn.model_selection import train_test_split

#用于模型评分
from sklearn.metrics import r2_score


# In[2]:


boston = datasets.load_boston()
train = boston.data
target = boston.target

#切割数据样本集合测试集
X_train,x_test,y_train,y_true = train_test_split(train,target,test_size=0.2)


# In[3]:

#创建学习模型
linear = LinearRegression()


# In[10]:

#训练模型
linear.fit(X_train,y_train)


# In[5]:

#预测数据
y_pre_linear = linear.predict(x_test)


# In[11]:

#评分
linear_score=r2_score(y_true,y_pre_linear)
display(linear_score)


# In[9]:

#可视化
#Linear
plt.plot(y_true,label='true')
plt.plot(y_pre_linear,label='linear')
plt.legend()

