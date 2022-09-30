# -*- coding: utf-8 -*-
"""
"""
#loading dataset
import pandas as pd
df = pd.read_csv("Zoo.csv")
df.shape
df.head()
df
# split as X and Y
Y = df["type"]
Y
X = df.iloc[:,1:17]
X
# standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_scale = SS.fit_transform(X)
X_scale

###############################################################################
#train and test spilt
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size=0.75,random_state=0)  # By default test_size=0.25


# model creation
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Prediction
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
print('Accuracy of KNN with K=5', accuracy_score(y_train, y_pred_train))
print('Accuracy of KNN with K=5', accuracy_score(y_test, y_pred_test))
print('Accuracy of KNN with K=5', confusion_matrix(y_test, y_pred_test))


# finding the best k value
est=[]
for i in range(1,15):
    knn=KNeighborsClassifier(n_neighbors=i)
    est.append(knn)
    score=[]
    for i in range(len(est)):
        est[i].fit(X_train,y_train)
        SS=est[i].fit(X_test,y_test)
        score.append(SS)
        score


#model visualization
from sklearn.model_selection import KFold,cross_val_score
knn=KNeighborsClassifier()
result=cross_val_score(knn,X,Y,cv=KFold(n_splits=5))
result.std()
result.mean()
#grid search
from sklearn.model_selection import GridSearchCV
para={ "n_neighbors":[3,4,5,7,9]}
grid=GridSearchCV(estimator=knn,param_grid=para,cv=KFold(n_splits=10))
grid.fit(X,Y)
grid.best_score_
grid.best_params_

###final model using kfold
final_model=KNeighborsClassifier(n_neighbors=3)
result=cross_val_score(final_model,X,Y,cv=KFold(n_splits=10))
result.mean()
result.min()
result.max()

#####final model using train and test
from sklearn.neighbors import KNeighborsClassifier
final_model=KNeighborsClassifier(n_neighbors=3)
final_model.fit(X_train,y_train)
#accuracy
from sklearn.metrics import accuracy_score
Y_pred=final_model.predict(X_test)
accuracy_score(y_test, Y_pred)
Y_pred1=final_model.predict(X_train)
accuracy_score(y_train, Y_pred1)





















