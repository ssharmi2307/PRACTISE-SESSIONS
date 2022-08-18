# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 19:29:34 2022

"""

import pandas as pd
df=pd.read_csv("Cutlets.csv")
df.shape
df.head()
from scipy import stats
zcalc ,pval = stats.ttest_ind(df["Unit A"],df["Unit B"])
print("zcal",zcalc.round(4))
print("pval",pval.round(4))
pval
if pval<0.05:
    print("reject null hypothesis, there is significant difference in the diameter of the cutlet between two units")
else:
    print("accept null hypothesis, there is no significant difference in the diameter of the cutlet between two units")



#######question2

import pandas as pd
df=pd.read_csv("LabTAT.csv")
df.shape
df.head()
from scipy import stats
p_value=stats.f_oneway(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],df.iloc[:,3])
p_value
if pval<0.05:
    print("reject null hypothesis,so there is difference in average TAT among the different laboratories")
else:
    print("accept null hypothesis,so there is no difference in average TAT among the different laboratories")




###### question3

import numpy as np
from scipy.stats import chi2_contingency
df= np.array([[50,142,131,70],[435,1523,1356,750]])
df# o/p is (Chi2 stats value, p_value, df, expected obsvations)
chi2_contingency(df)

p=0.6603094907091882
if p<0.05:
    print("Null hypothesis is rejected, not All proportions are equal")
else:
    print("Null hypothesis is accepted, All proportions are equal")



########question4

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
# load the dataset
data=pd.read_csv("Costomer+OrderForm.csv")
data
data.Phillippines.value_counts()
data.Phillippines.value_counts()
data.Indonesia.value_counts()
data.Malta.value_counts()
data.India.value_counts()
# Make a contingency table
obs=np.array([[271,267,269,280],[29,33,31,20]])
obs
# Chi2 contengency independence test
chi2_contingency(obs) # o/p is (Chi2 stats value, p_value, df, expected obsvations)

pval=0.2771
if pval< 0.05:
    print("test is proved,the defective %  varies by centre")
else:
    print("test is not proved,the defective % not varies by centre")

























