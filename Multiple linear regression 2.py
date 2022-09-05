# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#imporrting dataset
df=pd.read_csv("ToyotaCorolla.csv",encoding='latin1')
df.info()
df=pd.concat([df.iloc[:,2:4],df.iloc[:,6:7],df.iloc[:,8:9],df.iloc[:,12:14],df.iloc[:,15:18]],axis=1)
df
df[df.duplicated()]
df1=df.drop_duplicates().reset_index(drop=True)
df1
df1.describe()
df1.corr()
sns.pairplot(df1)
#("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]

# splitting the variables as X(independent variable) and Y(dependeent variable)
Y = df1["Price"] #dependent variable
#model 1
X = df1[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
#model 2
X = df1[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax"]]
#model 3
X = df1[["Age_08_04","KM","HP","cc","Doors","Gears"]]
#model 4
X = df1[["Age_08_04","KM","HP","cc","Doors"]]
#model 5
X = df1[["Age_08_04","KM","HP","cc"]]
#model 6
X = df1[["Age_08_04","KM","HP"]]
#model 7
X = df1[["Age_08_04","KM"]]
#model 8
X = df1[["Age_08_04"]]
#model  9
X = df1[["KM"]]
#model 10
X = df1[["HP"]]
#model 11
X = df1[["cc"]]
#model 12
X = df1[["Doors"]]


# Import Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X,Y)
model.coef_   ## To check the coefficients (B1)
model.intercept_  ## To check the Bo values
# Make predictions using independent variable values
Y_Pred=model.predict(X)
#cost fuction
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y,Y_Pred)
print("Mean square error: ", (mse).round(3))
r2 = r2_score(Y,Y_Pred)*100
print("R square: ", r2.round(3))


X1 = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
lm2 = sm.OLS(Y,X1).fit()

#the summary of our model
lm2.summary()

# parameters
lm2.params
lm2.pvalues
lm2.tvalues
lm2.rsquared_adj


RSS = np.sum((Y_Pred - Y)**2) # Residual sum of squares
Y_mean = np.mean(Y)
TSS = np.sum((Y - Y_mean)**2)
R2 = 1 - (RSS/TSS)
vif = 1/(1-R2)
print ("VIF value :",vif)

#Normal Q-Q plot of residuals"
sm.qqplot(lm2.resid,line='q')
plt.show()

#Test for Homoscedasticity or Heteroscedasticity
def standard_values(vals) : return (vals-vals.mean())/vals.std()
plt.scatter(standard_values(lm2.fittedvalues),standard_values(lm2.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show()

# Test for errors for independent variables
fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(lm2,'Age_08_04',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(lm2,'KM',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(lm2,'HP',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(lm2,'cc',fig=fig)
plt.show()


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(lm2,'Doors',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(lm2,'Gears',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(lm2,'Quarterly_Tax',fig=fig)
plt.show()


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(lm2,'Weight',fig=fig)
plt.show()


#New data for prediction
new_data=pd.DataFrame({'Age_08_04':23,"KM":50000,"HP":60,"CC":1500,"Doors":2,"Gears":4,"Quarterly_Tax":77,"Weight":83},index=[0])
model.predict(new_data)
pred_y=model.predict(X)
pred_y





