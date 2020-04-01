                     #Advanced HOUSE PRICE PREDICTIONS
            
                # PART 1 :- Getting the Data

#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the Dataset
train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')

                #PART 2:-  #Exploratory Data Analysis

#sales price info
info_train=train_df.SalePrice.describe()  

#to check skewness of target feature
print ("Skew is:", train_df.SalePrice.skew())  
plt.hist(train_df.SalePrice, color='blue')
plt.show()

#log transform the target variable since its skewed
target = np.log(train_df.SalePrice)   
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()

#get the numeric features from the dataset
numeric_features_train = train_df.select_dtypes(include=[np.number]) #include all numeric features
numeric_features_test = test_df.select_dtypes(include=[np.number])

#getting the categorical features and its description
categorical_features = train_df.select_dtypes(exclude=[np.number]) #exclude all numeric features
categorical_feature_description=categorical_features.describe()

#Check for missing values in train.csv and test.csv files
null_train = pd.DataFrame(numeric_features_train.isnull().sum().sort_values(ascending=False)[:80]) 
null_test = pd.DataFrame(numeric_features_test.isnull().sum().sort_values(ascending=False)[:80])

#Correlation matrix
corr_matrix=train_df.corr()  #.abs()
corr_sales=corr_matrix['SalePrice'].sort_values(ascending=False)[:38] #correlated features with SalesPrice

#Get the heatmap
import seaborn as sns
sns.heatmap(corr_matrix)

#Get the boxplot to identify an outlier
sns.boxplot(x=train_df['MSSubClass'])
sns.boxplot(x=train_df['LotFrontage'])


                   # PART 3-     Data Preprocessing

#Taking care of missing values
#train.columns.get_loc('LotFrontage')   #to get the location/index
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

train_df.iloc[:,[3,26,59]]=imputer.fit_transform(train_df.iloc[:,[3,26,59]])
train_df.select_dtypes(include=[np.number]).isnull().sum()

test_df.iloc[:,[3,26,34,36,37,38,48,47,59,62,61]]=imputer.fit_transform(test_df.iloc[:,[3,26,34,36,37,38,48,47,59,62,61]])
test_df.select_dtypes(include=[np.number]).isnull().sum()

#Removing Outliers
from scipy import stats
def drop_numerical_outliers(numeric_features_train, z_thresh=3):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = numeric_features_train.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh, result_type='reduce') \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    numeric_features_train.drop(numeric_features_train.index[~constrains], inplace=True)

drop_numerical_outliers(train_df)
#drop_numerical_outliers(test_df)

#Taking care of categorical data (Encoding)

#OneHotEncoding approach is not appropriate for multiple linear regression.

#1 Encoding-MSZoning
print (train_df.MSZoning.value_counts())
def encode(x):        #Encoding RL as 1 and others as 0.
    return 1 if x == 'RL' else 0  #to encode
train_df['enc_MSZoning'] = train_df.MSZoning.apply(encode)
test_df['enc_MSZoning'] = test_df.MSZoning.apply(encode)
print (train_df.enc_MSZoning.value_counts())
  
#2 - Encoding Street
print (train_df.Street.value_counts())
def encode(x):        #Encoding RL as 1 and others as 0.
    return 1 if x == 'Pave' else 0  #to encode
train_df['enc_Street'] = train_df.Street.apply(encode)
test_df['enc_Street'] = test_df.Street.apply(encode)
print (train_df.enc_Street.value_counts())

#3 - Encoding LotShape
print (train_df.LotShape.value_counts())
def encode(x):
    return 1 if x == 'Reg' else 0  #to encode
train_df['enc_LotShape'] = train_df.LotShape.apply(encode)
test_df['enc_LotShape'] = test_df.LotShape.apply(encode)
print (train_df.enc_LotShape.value_counts())

#4 -Encoding HouseStyle
print (train_df.HouseStyle.value_counts())
def encode(x):
    return 0 if x == '1Story' else 1  #to encode
train_df['enc_HouseStyle'] = train_df.HouseStyle.apply(encode)
test_df['enc_HouseStyle'] = test_df.HouseStyle.apply(encode)
print (train_df.enc_HouseStyle.value_counts())

#6 -Encoding GarageCond
print (train_df.GarageCond.value_counts())
def encode(x):
    return 1 if x == 'TA' else 0  #to encode
train_df['enc_GarageCond'] = train_df.GarageCond.apply(encode)
test_df['enc_GarageCond'] = test_df.GarageCond.apply(encode)
print (train_df.enc_GarageCond.value_counts())

#7- Encoding Central Air           #when only 2 categories are present
print ("Original: \n")
print (train_df.CentralAir.value_counts(), "\n")
train_df['enc_CentralAir'] = pd.get_dummies(train_df.CentralAir, drop_first=True)
test_df['enc_CentralAir'] = pd.get_dummies(test_df.CentralAir, drop_first=True)
print ('Encoded: \n')
print (train_df.enc_CentralAir.value_counts())

"""#Not the best method to take care of missing values
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0) #Check if the all of the columns have 0 null values."""

#Remove highly correlated features
corr_matrix=train_df.corr()  #.abs()
#Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
#since every correlation matrix is symmetric 
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)] #List Comprehension
print(to_drop)

# Drop Marked Features
train_df=train_df.drop(train_df[to_drop], axis=1)

#get the final numeric features from the dataset for modelling
train = train_df.select_dtypes(include=[np.number]) 
test = test_df.select_dtypes(include=[np.number])


                    #PART 4 -  Build a linear model
#DV and IDV's
y = np.log(train.SalePrice)   
#y=train.iloc[:,-7].values
X = train.drop(['SalePrice', 'Id'], axis=1)

#Building the optiomal model using Backward Elimination
#import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm
X=np.append(arr=np.ones((1460,1)).astype(int),values=X,axis=1)
#np.ones create 1 column with only 1's
#axis=1 for column,0 for rows

#actual backward elimination
#Compare x and x_opt index 
X_opt=X[:,0:43]
regressor_OLS=sm.OLS(endog=y ,exog=X_opt).fit()
regressor_OLS.summary()

"""X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,
           11,12,13,14,15,16,17,18,19,20,
           21,22,23,24,25,26,27,28,29,30,
           31,32,33,34,35,36,37,38,39,40,
           41,42]]
regressor_OLS=sm.OLS(endog=y ,exog=X_opt).fit()
regressor_OLS.summary()"""

X_opt=X[:,[0,1,2,3,4,5,6,7,9,
           12,13,14,17,18,19,
           22,24,26,29,
           32,33,35,38,
           41,42]]
regressor_OLS=sm.OLS(endog=y ,exog=X_opt).fit()
regressor_OLS.summary()

"""X_opt=X[:,[0,1,2,3,4,5,6,7,
           11,12,16,17,18,19,
           22,24,26,29,
           32,33,38,
           41,42]]
regressor_OLS=sm.OLS(endog=y ,exog=X_opt).fit()
regressor_OLS.summary()"""
#removed 20 variables(p>0.05) by backward elimination
#The higher the t-statistic (and the lower the p-value), the more significant the predictor

#Final best variables for modelling
X=train.iloc[:,[0,1,2,3,4,5,6,7,9,
           12,13,14,17,18,19,
           22,24,26,29,
           32,33,35,38,
           41,42]].values
y=y  

#Splitting the dataset into trining set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

"""# Feature Scaling     #FS is required in Dimension reduction
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression           
regressor = LinearRegression()                              
model=regressor.fit(X_train, y_train) 

"""import xgboost
from xgboost import XGBRegressor
regression = XGBRegressor()
model=regression.fit(X_train, y_train)"""

#Predicting the test set results
y_pred=regressor.predict(X_test)   #using train set

# Applying k-Fold Cross Validation (model evaluation)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std() 

#Evaluate the model
print ("R^2 is: \n", model.score(X_test, y_test))
#higher r-squared value means a better fit.

#RMSE measures the distance between our predicted values and actual values.
from sklearn import metrics
print(metrics.mean_absolute_error(y_test,y_pred))   #MAE
print(metrics.mean_squared_error(y_test,y_pred))   #MSE
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))   #RMSE

#Plot
actual_values = y_test
plt.scatter(y_pred, actual_values, alpha=.7,color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()

"""
#Regularization
for i in range (-2, 3):
    alpha = 10**i
    rm = model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()
#adjusting the alpha did not substantially improve our model
"""
#Predicting the test set results
#Predicting the test set results
features = test.select_dtypes(include=[np.number]).drop(['Id','MasVnrArea',
                             'BsmtFinSF2','BsmtUnfSF','LowQualFinSF','GrLivArea',
                             'HalfBath','BedroomAbvGr',
                             'TotRmsAbvGrd','GarageYrBlt','GarageArea',
                             'WoodDeckSF','PoolArea', #poolarea/3ssnporch
                             'ScreenPorch','MiscVal','YrSold',
                             'enc_MSZoning','enc_LotShape','enc_HouseStyle'], axis=1)


features=np.append(arr=np.ones((1459,1)).astype(int),values=features,axis=1) 
#since i have added one column of 1's during backward elimination
predictions = model.predict(features)
final_predictions = np.exp(predictions)

#Getting a csv file
output=pd.DataFrame({'Id':test.Id, 'SalePrice':final_predictions})
output.to_csv('my_submission_FS10.csv', index=False)