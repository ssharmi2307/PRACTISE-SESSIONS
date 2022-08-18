# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 16:41:24 2022

@author: Gopinath
"""


#######question 1



import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
# Loading the data
df = pd.read_csv("delivery_time.csv")
df.shape

# scatter plot
df.plot.scatter(x='Sorting Time', y='Delivery Time')
df=df.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
df
## correlation analysis
df.corr()
## data visualization
sns.regplot(x=df['sorting_time'],y=df['delivery_time'])
## model fitting
model=smf.ols("delivery_time~sorting_time",data=df).fit()
# Finding Coefficient parameters
model.params
# Finding tvalues and pvalues
model.tvalues , model.pvalues
# Finding Rsquared Values
model.rsquared , model.rsquared_adj
## model prediction
new_data=df.iloc[:,1]
new_data
data_pred=pd.DataFrame(new_data,columns=['sorting_time'])
data_pred
model.predict(data_pred)




######question 2


import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
# Loading the data
df = pd.read_csv("Salary_Data.csv")
df.shape

# Data visualization
df.plot.scatter(x='YearsExperience', y='Salary')
df
# correlation analysis
df.corr()
# model fitting
model=smf.ols("Salary~YearsExperience",data=df).fit()
# Finding Coefficient parameters
model.params
# Finding tvalues and pvalues
model.tvalues , model.pvalues
# Finding Rsquared Values
model.rsquared , model.rsquared_adj

# model prediction
new_data=df.iloc[:,0]
new_data
data_pred=pd.DataFrame(new_data,columns=['YearsExperience'])
data_pred
model.predict(data_pred)












