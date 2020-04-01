
                                #HOUSE PRICE PREDICTIONS
# PART 1 :- 
                        # PART 1- Getting the Data

#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the Dataset
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#PART 2:- 
                        #PART 2- Exploratory Data Analysis

#sales price info
info_train=train.SalePrice.describe()  

#to plot the skewness of sales price
print ("Skew is:", train.SalePrice.skew())  
plt.hist(train.SalePrice, color='blue')
plt.show()

#log transform the target variable since its skewed
target_y = np.log(train.SalePrice)  
print ("Skew is:", target_y.skew())
plt.hist(target_y, color='blue')
plt.show()

#get the numeric features from the dataset
numeric_features_train = train.select_dtypes(include=[np.number]) 
numeric_features_test = test.select_dtypes(include=[np.number])
 
#getting the categorical features and its description
categorical_features = train.select_dtypes(exclude=[np.number]) #exclude all numeric features
categorical_feature_description=categorical_features.describe()

#correlation matrix
corr = numeric_features_train.corr()    
print(corr['SalePrice'].sort_values(ascending=False)[:38], '\n') #38 most +vely correlated features with SalesPrice
#print(corr['SalePrice'].sort_values(ascending=False)[-10:])  #10 most -vely correlated

#Getting the heatmap
import seaborn as sns
sns.heatmap(corr)

#remove one of two features that have a correlation higher than 0.9
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = numeric_features_train.columns[columns]
train_corr = numeric_features_train[selected_columns]
#dataset(train_corr) has only those columns with correlation less than 0.9

#Getting the numeric features with null values
nulls_train = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:80]) #features with null values
nulls_train.columns = ['Null Count']
nulls_train.index.name = 'Feature'
nulls_test = pd.DataFrame(test.isnull().sum().sort_values(ascending=False)[:80]) #features with null values
nulls_test.columns = ['Null Count']
nulls_test.index.name = 'Feature'

                   # PART 3-     Data Preprocessing

#Taking care of missing data
num_null_train = pd.DataFrame(numeric_features_train.isnull().sum().sort_values(ascending=False)[:80]) #features with null values
num_null_test = pd.DataFrame(numeric_features_test.isnull().sum().sort_values(ascending=False)[:80]) #features with null values

#Taking care of categorical data
                   
#1 -MSZoning 
print ("Original: \n")
print (train.MSZoning.value_counts(), "\n")  #Counts
def encode(x):        #Encoding RL as 1 and others as 0.
    return 1 if x == 'RL' else 0  #to encode
train['enc_MSZoning'] = train.MSZoning.apply(encode)
test['enc_MSZoning'] = test.MSZoning.apply(encode)
print (train.enc_MSZoning.value_counts())   #Check encoded value
#to check barplot
condition_pivot = train.pivot_table(index='enc_MSZoning', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()  

# 2- Street
print ("Original: \n")
print (train.Street.value_counts(), "\n")
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(test.Street, drop_first=True)
print ('Encoded: \n')
print (train.enc_street.value_counts())

#3 -GarageCond
print (train.GarageCond.value_counts())
def encode(x):
 return 1 if x == 'TA' else 0  #to encode
train['enc_GarageCond'] = train.GarageCond.apply(encode)
test['enc_GarageCond'] = test.GarageCond.apply(encode)
print (train.enc_GarageCond.value_counts())


# 4- Central Air           #when only 2 categories are present
print ("Original: \n")
print (train.CentralAir.value_counts(), "\n")
train['enc_CentralAir'] = pd.get_dummies(train.CentralAir, drop_first=True)
test['enc_CentralAir'] = pd.get_dummies(test.CentralAir, drop_first=True)
print ('Encoded: \n')
print (train.enc_CentralAir.value_counts())



train_corr = train.corr()    #correlation between numerical features and target.
print(train_corr['SalePrice'].sort_values(ascending=False)[:15], '\n') #10 most +vely correlated features with SalesPrice
print(train_corr['SalePrice'].sort_values(ascending=False)[-10:]) 


# PART 4 Bulding a linear model

#DV and IDV features
"""X=train.iloc[:,[17,46,61,62,38,43,49,54,19,20,59,26,56,34]].values
#Both are very correlated (38,43)
19,20,59 very correlated 
59-Nan
26-Nan
"""
#Not the best method to take care of missing values
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0) #Check if the all of the columns have 0 null values.

"""
X=train.iloc[:,[17,46,61,62,38,49,54,19,26,56,34]].values
y=train.iloc[:,-5].values

#Missing values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
X[:,8:9]=imputer.fit_transform(X[:,8:9])
"""

X = data.drop(['SalePrice', 'Id'], axis=1)   #From train.csv
y = np.log(train.SalePrice)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.25)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression           
regressor = LinearRegression()                              
model=regressor.fit(X_train, y_train) 

#Predicting the test set results
y_pred=regressor.predict(X_test)   #using train set

# Applying k-Fold Cross Validation (model evaluation)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std() 

# PART 4 Evaluate the model
print ("R^2 is: \n", model.score(X_test, y_test))

from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, y_pred))
#RMSE measures the distance between our predicted values and actual values.

actual_values = y_test
plt.scatter(y_pred, actual_values, alpha=.7,color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()

#Predicting the test.csv results
"""features = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
features_X=features.iloc[:,[3,15,25,26,11,18,22,5,7,23,8]]
predictions = model.predict(features_X)
final_predictions = np.exp(predictions)"""
#Getting the results on test.csv file
test_features = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(test_features)    #using test 
final_predictions = np.exp(predictions)

#compare the result
print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])

#Getting a csv file
output=pd.DataFrame({'Id':test.Id, 'SalePrice':final_predictions})
output.to_csv('my_submission_SLR.csv', index=False)