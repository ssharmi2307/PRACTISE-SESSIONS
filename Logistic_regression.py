# -*- coding: utf-8 -*-
"""

@author: Gopinath
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
df=pd.read_csv('bank-full.csv',sep=";")
df.head()
df.info()
df.shape
df


# Custom Binary Encoding of Binary o/p variables
df['default'] = np.where(df['default'].str.contains("yes"), 1, 0)
df['housing'] = np.where(df['housing'].str.contains("yes"), 1, 0)
df['loan'] = np.where(df['loan'].str.contains("yes"), 1, 0)
df['y'] = np.where(df['y'].str.contains("yes"), 1, 0)
df.info()

# One-Hot Encoding of categrical variables
df=pd.get_dummies(df,columns=['job','marital','education','contact','poutcome','month'])
df
pd.set_option("display.max.columns", None)
df

#target and feature

x=pd.concat([df.iloc[:,0:10],df.iloc[:,11:]],axis=1)
y=df.iloc[:,10]

#standardization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(x)

#logistic regression model creation
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x,y)
Y_pred=model.predict(X)
Y_pred
data=pd.DataFrame({"Actual":y,"Predicted":Y_pred})
data


#model validation
from sklearn.metrics import classification_report
classification_report(y, Y_pred)


from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y,Y_pred)
accuracy_score(y, Y_pred)

#model visualization
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
fpr,tpr,thresh=roc_curve(y,Y_pred)
auc(fpr,tpr)
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],"--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend
plt.show()





