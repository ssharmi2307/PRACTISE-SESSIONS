# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 20:58:35 2022

"""
#qusetion 1
from scipy.stats import norm
nd = norm(45,8) # mean, sd
1-nd.cdf(50)

#qusetion2
from scipy.stats import norm
nd = norm(38,6) # mean, sd

# 	a. employees at the processing center are older than 44
1- nd.cdf(44)
    #between 38 and 44
nd.cdf(44)-nd.cdf(38)


# 	b. training program for employees under the age of 30
nd.cdf(30)
#the expected to attract about 36 employees
0.09121121972586788*400 ## no of employees with prob 0.0912% being under age of 30




######question5
from scipy.stats import norm
from scipy import stats
# Rupee range (centered on the mean)that contains 95% probability
nd = norm(12*45,5*45)  # mean= mean1+mean2, sd^2= sd1^2+sd2^2
stats.norm.interval(0.95,12*45,5*45) #in million rupees
# b.	5th percentile of profit (in Rupees) for the company
#from the z table 5 percentile = -1.645, so X=μ + Zσ
x=540+(-1.645)*(225)
x #in million rupees
# c.P(X<0) in division 1
stats.norm.cdf(0,5,3)
#.P(X<0) in division 2
stats.norm.cdf(0,7,4)
#Division1 has a larger probability of making a loss in a given year


x3 = nd.cdf(70000)
x4 = nd.cdf(60000)
x3-x4



















