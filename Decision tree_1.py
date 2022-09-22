# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:29:55 2022


"""
import warnings
warnings.filterwarnings('ignore')
#loading dataset
import pandas as pd
df=pd.read_csv("Company_Data.csv")
df
df.shape

import numpy as np
df['Urban']=np.where(df['Urban'].str.contains("Yes"),1,0)
df['US']=np.where(df['US'].str.contains("Yes"),1,0)
df['ShelveLoc']=df['ShelveLoc'].map({'Good':1,'Medium':2,'Bad':3})
df=df.assign(Sale=pd.cut(df['Sales'],bins=[ 0, 4, 9,15],labels=['Low', 'Medium', 'High']))
df
#converting target variable in categorical form using Label encoder
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['Sale'] = label_encoder.fit_transform(df['Sale'])
df.head()
df
df1 = df.drop('Sales', axis =1)
df1

#plotting pair plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(df,hue = "Sale")
# correlation matrix
sns.heatmap(df.corr())

#traget and features
X = df1.iloc[:,0:10]
list(X)
Y = df['Sale']
Y

#train and test splitting
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.75,random_state=0)

#model creation
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion="entropy",ccp_alpha=0.0,max_depth=3)
model.fit(X_train,Y_train)

#accuracy
from sklearn.metrics import accuracy_score
Y_pred=model.predict(X_test)
accuracy_score(Y_test, Y_pred)
Y_pred1=model.predict(X_train)
accuracy_score(Y_train, Y_pred1)

#model validation
import matplotlib.pyplot as plt
from sklearn import tree
#PLot the decision tree
plt.figure(figsize=(25,15))
tree.plot_tree(model);
plt.show()

#exporting tree structure to text
from sklearn.tree import export_text
r = export_text(model,feature_names=['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','Sale'])
 'Education',
 'Urban',
 'US'','Sales'])
print(r)

path=model.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas,impurities=path.ccp_alphas,path.impurities
ccp_alphas

clf=[]
for i in ccp_alphas:
    test_model=DecisionTreeClassifier(criterion='entropy',random_state=0,ccp_alpha=i)
    test_model.fit(X_train,Y_train)
    clf.append(test_model)
clf

#comparing testing and training data
train_score=[]
test_score=[]
for mod in clf:
    train_score.append(mod.score(X_train,Y_train))
    test_score.append(mod.score(X_test,Y_test))
    train_score
    test_score

#to check accuracy
plt.scatter(ccp_alphas,train_score,label="training",color='red')
plt.scatter(ccp_alphas,test_score,label="training",color='green')
plt.plot(ccp_alphas,train_score,label="training",color='red')
plt.plot(ccp_alphas,test_score,label="training",color='green')
plt.legend()
plt.grid()
plt.xlim(0,0.043)
plt.show()

#final model
from sklearn.tree import DecisionTreeClassifier
final_model=DecisionTreeClassifier(criterion="entropy",ccp_alpha=0.043,max_depth=3)
final_model.fit(X_train,Y_train)

#accuracy
from sklearn.metrics import accuracy_score
Y_pred=final_model.predict(X_test)
accuracy_score(Y_test, Y_pred)
Y_pred1=final_model.predict(X_train)
accuracy_score(Y_train, Y_pred1)



