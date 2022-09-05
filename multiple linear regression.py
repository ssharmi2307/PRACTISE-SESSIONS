# -*- coding: utf-8 -*-

# import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# import dataset
df=pd.read_csv("50_Startups.csv")
df
df.info()
#correlation analysis
df.corr()

#sns plot
sns.pairplot(df)

# splitting the variables as X(independent variable) and Y(dependeent variable)
Y = df["Profit"] #dependent variable
#model 1
X = df[["R&D Spend","Administration","Marketing Spend"]]
#model 2
X = df[["R&D Spend","Administration"]]
#model 3
X = df[["R&D Spend","Marketing Spend"]]
#model 4
X = df[["Administration","Marketing Spend"]]
#model 5
X = df[["R&D Spend"]]
#model 6
X = df[["Administration"]]
#model 7
X = df[["Marketing Spend"]]


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
sm.graphics.plot_regress_exog(lm2,'R&D Spend',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(lm2,'Administration',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(lm2,'Marketing Spend',fig=fig)
plt.show()


#New data for prediction
new_data=pd.DataFrame({'R&D Spend':8000,"Administration":10000,"Marketing Spend":170000},index=[0])
model.predict(new_data)
pred_y=model.predict(X)
pred_y














