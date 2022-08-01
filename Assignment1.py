# -*- coding: utf-8 -*-
"""
"""

####### question 7

import pandas as pd
df= pd.read_csv("Q7.csv")
df.shape
list(df)
 #mean, stddev,range
df["Points"].describe()
df["Score"].describe()
df["Weigh"].describe()
#median
df["Points"].median()
df["Score"].median()
df["Weigh"].median()
#mode
df["Points"].mode()
df["Score"].mode()
df["Weigh"].mode()
#variance
df["Points"].var()
df["Score"].var()
df["Weigh"].var()
#boxplot and outliers
df.boxplot(column=['Points','Score','Weigh'],vert=True)

######### question 8
import numpy as np
x1 = np.array([108, 110, 123, 134, 135, 145, 167, 187, 199])
x1.mean()

######### question 9

import pandas as pd
df= pd.read_csv("Q9_a.csv")
df.shape
list(df)
from scipy.stats import skew
skew(df['speed'])
skew(df['dist'])

from scipy.stats import kurtosis
kurtosis(df['speed'])
kurtosis(df['dist'])
df.boxplot(column=['speed','dist'],vert=True)


import pandas as pd
df= pd.read_csv("Q9_b.csv")
df.shape
list(df)
from scipy.stats import skew
skew(df['SP'])
skew(df['WT'])

from scipy.stats import kurtosis
kurtosis(df['SP'])
kurtosis(df['WT'])
df.boxplot(column=['SP','WT'],vert=True)


####### question 20
import pandas as pd
df=pd.read_csv("Cars.csv")
list(df)
df["MPG"].mean()
df["MPG"].std()
from scipy.stats import norm 
nd=norm(34.422,9.131)#mean,stddev
# a)
1-nd.cdf(38)
# b)
nd.cdf(40)
# c)
nd.cdf(50)-nd.cdf(20)

####### question 21
# a)
import pandas as pd
df=pd.read_csv("Cars.csv")
import pylab          
import scipy.stats as st
st.probplot(df['MPG'], dist="norm",plot=pylab) ### qq plot for MPG


# b)
import pandas as pd
import pylab          
import scipy.stats as st
df= pd.read_csv("wc-at.csv")

st.probplot(df['Waist'], dist="norm",plot=pylab) ##qq plot for Waist


st.probplot(df['AT'],dist="norm",plot=pylab) #qq plot fot AT



###### question 24
from scipy.stats import norm 
nd=norm(270,90)#mean,stddev
nd.cdf(260)










