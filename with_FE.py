#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:55:30 2020

@author: pranavkalikate
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:32:40 2020

@author: pranavkalikate
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 19:13:28 2020

@author: pranavkalikate
"""

                                    #HOUSE PRICE PREDICTIONS
# PART 1 :- Getting the Data

#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the Dataset
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#PART 2:- 

#Exploratory Data Analysis
info_train=train.SalePrice.describe()  #sales price info

print ("Skew is:", train.SalePrice.skew())  #to check skewness of sales price
plt.hist(train.SalePrice, color='blue')
plt.show()

target = np.log(train.SalePrice)   #log transform the target variable since its skewed
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()

numeric_features = train.select_dtypes(include=[np.number]) #get the numeric features from the dataset
numeric_features.dtypes

corr = numeric_features.corr()    #correlation between numerical features and target.
print(corr['SalePrice'].sort_values(ascending=False)[:10], '\n') #10 most +vely correlated features with SalesPrice
print(corr['SalePrice'].sort_values(ascending=False)[-10:])  #10 most -vely correlated

#First 7 +vely correlated features plots
plt.scatter(x=train['OverallQual'], y=target)  
plt.ylabel('Sale Price')
plt.xlabel('Overall Qual')
plt.show()

plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()

train = train[train['GrLivArea'] < 4000]    #Removing outliers from GrLivArea
plt.scatter(x=train['GrLivArea'], y=np.log(train.SalePrice))
plt.xlim(-800,4500) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()

plt.scatter(x=train['GarageCars'], y=np.log(train.SalePrice))  
plt.ylabel('Sale Price')
plt.xlabel('Garage Cars')
plt.show()

plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))  
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()

train = train[train['GarageArea'] < 1200]    #Removing outliers from GarageArea
plt.scatter(x=train['GrLivArea'], y=np.log(train.SalePrice))
plt.xlim(-100,4000) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()

plt.scatter(x=train['TotalBsmtSF'], y=np.log(train.SalePrice))  
plt.ylabel('Sale Price')
plt.xlabel('TotalBsmtSF')
plt.show()

"""train = train[train['TotalBsmtSF'] < 2500]    #Removing outliers from TotalBsmtSF
plt.scatter(x=train['TotalBsmtSF'], y=np.log(train.SalePrice))
plt.xlim(-200,3000) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('TotalBsmtSF')
plt.show()"""

plt.scatter(x=train['1stFlrSF'], y=np.log(train.SalePrice))  
plt.ylabel('Sale Price')
plt.xlabel('1stFlrSF ')
plt.show()

plt.scatter(x=train['FullBath'], y=np.log(train.SalePrice))
plt.ylabel('Sale Price')
plt.xlabel('FullBath')
plt.show()

#Getting the nemeric features with null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25]) #features with null values
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls

#getting the non-numeric (categorical features) and its description
categorical_features = train.select_dtypes(exclude=[np.number]) #exclude all numeric features
categorical_feature_description=categorical_features.describe()

#PART 3 Pre-processing
"""
#Missing values
nan_values=numeric_features.isnull().sum().sort_values(ascending=False)
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
#features with missing_values not of interest as well as consists of lot of missing values
#better to omit those features

#Not the best method to take care of missing values
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0) #Check if the all of the columns have 0 null values.
"""

#Taking care of categorical features
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
"""# For Street feature
ct = ColumnTransformer([('encoder', OneHotEncoder(), [5])], remainder='passthrough')
train = np.array(ct.fit_transform(train))
test=np.array(ct.fit_transform(test))
train = train[:, 1:]
test=test[:, 1:]"""

#1 -MSZoning
condition_pivot = train.pivot_table(index='MSZoning', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('MSZoning')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()  #to plot 
def encode(x):        #Encoding RL as 1 and others as 0.
 return 1 if x == 'RL' else 0  #to encode
train['enc_MSZoning'] = train.MSZoning.apply(encode)
test['enc_MSZoning'] = test.MSZoning.apply(encode)

print (train.enc_MSZoning.value_counts())

condition_pivot = train.pivot_table(index='enc_MSZoning', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()  #to check barplot

#3 -GarageCond
condition_pivot = train.pivot_table(index='GarageCond', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('GarageCond')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()  #to plot before encoding
def encode(x):
 return 1 if x == 'TA' else 0  #to encode
train['enc_GarageCond'] = train.GarageCond.apply(encode)
test['enc_GarageCond'] = test.GarageCond.apply(encode)
print (train.enc_GarageCond.value_counts())
condition_pivot = train.pivot_table(index='enc_GarageCond', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()    #check barplot after encoding


# 4- Central Air           #when only 2 categories are present
print ("Original: \n")
print (train.CentralAir.value_counts(), "\n")
train['enc_CentralAir'] = pd.get_dummies(train.CentralAir, drop_first=True)
test['enc_CentralAir'] = pd.get_dummies(test.CentralAir, drop_first=True)
print ('Encoded: \n')
print (train.enc_CentralAir.value_counts())

# 5- Street
print ("Original: \n")
print (train.Street.value_counts(), "\n")
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(test.Street, drop_first=True)
print ('Encoded: \n')
print (train.enc_street.value_counts())

train_corr = train.corr()    #correlation between numerical features and target.
print(train_corr['SalePrice'].sort_values(ascending=False)[:15], '\n') #10 most +vely correlated features with SalesPrice
print(train_corr['SalePrice'].sort_values(ascending=False)[-10:]) 

#Building
X=train.iloc[:,[17,46,61,62,38,43,49,54,19,20,56,34]].values
y=np.log(train.SalePrice)
#19,20,59 very correlated 
#59-Nan
#26-Nan

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.25)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)  
model=regressor.fit(X, y)

#Predicting the test set results
y_pred=regressor.predict(X_test)   #using train set

# Applying k-Fold Cross Validation (model evaluation)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std() 

#Evaluate the model
print ("R^2 is: \n", model.score(X_test, y_test))

from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, y_pred))

actual_values = y_test
plt.scatter(y_pred, actual_values, alpha=.7,color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()

features = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
features_X=features.iloc[:,[3,15,25,26,11,12,18,22,5,6,23,8]]
predictions = model.predict(features_X)
final_predictions = np.exp(predictions)

#Getting a csv file
output=pd.DataFrame({'Id':test.Id, 'SalePrice':final_predictions})
output.to_csv('submission_1.csv', index=False)