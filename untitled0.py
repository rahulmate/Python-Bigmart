# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:12:57 2018

@author: admin
"""

import pandas as pd
import numpy as np
import math
import pylab

# uses Ordinary Least Squares (OLS) method
# -------------------------------------------
import statsmodels.api as sm

from sklearn.cross_validation import train_test_split
import scipy.stats as stats

import seaborn as sns

# VIF
# ---
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Feature selection
# -----------------
from sklearn.feature_selection import f_regression as fs
# read the input file
# --------------------
path="D:\\python project\\salesdata.csv"
sales = pd.read_csv(path)
sales.head(100)
sales.shape
sales.dtypes
# summarize the dataset
# clearer view. removed the 1st row as it contains same info (total records)
# ------------------------------------------------------------
desc = sales.describe()
desc = desc.drop(desc.index[0]) # dropping the record count
desc
# Get all the factor X-variables
# --------------------------------------
factor_x = sales.select_dtypes(exclude=["int64","float64","category"]).columns.values
print(factor_x)
# Unique values of all Factor variables
# --------------------------------------
for c in factor_x:
    print("Factor variable = '" + c + "'")
    print(sales[c].unique())
    print("***")
    
#IN Item_Fat_Content LF must be Low Fat and reg is Regular
sales.Item_Fat_Content[sales.Item_Fat_Content=='reg']='Regular'
sales.Item_Fat_Content[
        (sales.Item_Fat_Content == 'low fat') | 
        (sales.Item_Fat_Content == 'LF')
        ]='Low Fat'

cols = list(sales.columns)
type(cols)
cols.remove("Item_Outlet_Sales")
print(cols)

for c in cols:
    if (len(sales[c][sales[c].isnull()])) > 0:
        print("WARNING: Column '{}' has NULL values".format(c))

    if (len(sales[c][sales[c] == 0])) > 0:
        print("WARNING: Column '{}' has value = 0".format(c))
        
#Determine the average weight per item:
item_avg_weight = sales.pivot_table(values='Item_Weight', index='Item_Identifier')

#Get a boolean variable specifying missing Item_Weight values
miss_bool = sales['Item_Weight'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
print (sum(miss_bool))
sales.Item_Weight[sales.Item_Weight == 0] = np.nan
means_wt = np.nanmean(sales.Item_Weight)
sales.Item_Visibility[sales.Item_Visibility == 0] = np.nan
means_vi = np.nanmean(sales.Item_Visibility)
sales.Item_Weight = sales.Item_Weight.fillna(means_wt)
sales.Item_Visibility = sales.Item_Visibility.fillna(means_vi)


l=sales.filter(["Outlet_Size", "Outlet_Location_Type"]).mode()
sales[["Outlet_Size", "Outlet_Location_Type"]]=sales[["Outlet_Size", "Outlet_Location_Type"]].fillna(value=l.iloc[0])

#Get the first two characters of ID:
sales['Item_Identity'] = sales['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
sales['Item_Identity'] = sales['Item_Identity'].map({'FD':'Food',
                                                         'NC':'Non-Consumable',
                                                         'DR':'Drinks'})
sales.head(5)
sales['Outlet_Year'] = 2018 - sales['Outlet_Establishment_Year']
del sales['Item_Identifier']
del sales['Outlet_Establishment_Year']
#DUMMY Variable
factor_x = sales.select_dtypes(exclude=["int64","float64","category"]).columns.values
print(factor_x)
for var in factor_x:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(sales[var], prefix=var)
    data1=sales.join(cat_list)
    sales = data1

# old+dummy columns
sales.columns

sales.head()

sales_vars=sales.columns.values.tolist()
to_keep = [i for i in sales_vars if i not in factor_x]
print(to_keep)
len(to_keep)

# create the final dataset with the final columns set
# ---------------------------------------------------
sales_final = sales[to_keep]
sales_final.columns
sales_final['Outlet_Sales']=sales_final['Item_Outlet_Sales']
del sales_final['Item_Outlet_Sales']
###############################################################################
train1=sales_final.iloc[0:8523,:]
test1=sales_final.iloc[8524:14205,:]
test1.shape
train1.shape
train, test = train_test_split(train1, test_size = 0.3)
print(train.shape)
print(test.shape)
# split the dataset into train and test
# --------------------------------------
# split the train and test into X and Y variables
# ------------------------------------------------
train_x = train.iloc[:,0:45]; train_y = train.iloc[:,45]
test_x  = test.iloc[:,0:45];  test_y = test.iloc[:,45]
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
train.dtypes
test1.head(5)
# function -> getresiduals()
# -------------------------------
def getresiduals(lm,train_x,train_y):
    predicted = lm.predict(train_x)
    actual = train_y
    residual = actual-predicted
    return(residual)
def SSR(lm,train_x,train_y):
    predicted = lm.predict(train_x)
    mean_val =np.mean(train_y)
    SSR = (train_y - mean_val)**2
    return(SSR)
import random as r
train_x = sm.add_constant(train_x)
test_x = sm.add_constant(test_x)
lm1 = sm.OLS(train_y, train_x).fit()
lm1.summary()
lm1.params
residuals = getresiduals(lm1,train_x,train_y)
print(sum(residuals))
SSR=SSR(lm1,train_x,train_y)
print(sum(SSR))
R=1-(sum(residuals)/sum(SSR))
AdjustedR=1-((1-R)*(5966-1)/(5966-45-1))
print(residuals.mean())
y = lm1.predict(train_x)
sns.set(style="whitegrid")
sns.residplot(residuals,y,lowess=True,color="g")
stats.probplot(residuals,dist="norm",plot=pylab)
pylab.show()

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(train_x.values, i)
for i in range(train_x.shape[1])]
vif["features"] = train_x.columns
print(vif)

train.head()
X=train_x.iloc[:,1:45]
features = fs(X,train_y,center=True)
list(features[0])
# pd.DataFrame({'column':cols[1:9], 'coefficieint':coefficients})

df_features = pd.DataFrame({"columns":train_x.columns[1:45], 
                            "score":features[0],
                            "p-val":features[1]
                            })
print(df_features)

pdct1 = lm1.predict(test_x)
print(pdct1)

# mean square error
# -----------------
mse = np.mean((pdct1 - test_y)**2)
print("MSE = {0}, RMSE = {1}".format(mse,math.sqrt(mse)))

actual = list(test_y.head(50))
predicted = np.round(np.array(list(pdct1.head(50))),2)
print(predicted)

df_results = pd.DataFrame({'actual':actual, 'predicted':predicted})
print(df_results)
################################################Selecting important features
new_train=train1[['Item_MRP','Outlet_Identifier_OUT010','Outlet_Year','Item_Fat_Content_Low Fat','Item_Fat_Content_Regular','Outlet_Identifier_OUT019','Outlet_Identifier_OUT027','Outlet_Type_Grocery Store', 'Outlet_Type_Supermarket Type3','Outlet_Identifier_OUT013','Outlet_Identifier_OUT035','Outlet_Size_High', 'Outlet_Size_Medium', 'Outlet_Size_Small','Outlet_Location_Type_Tier 1', 'Outlet_Type_Supermarket Type1','Outlet_Sales' ] ]
new_train.dtypes
new_train.shape
train, test = train_test_split(new_train, test_size = 0.3)
train_x = train.iloc[:,0:16]; train_y = train.iloc[:,16]
test_x  = test.iloc[:,0:16];  test_y = test.iloc[:,16]
lm2 = sm.OLS(train_y, train_x).fit()
lm2.summary()
residuals = getresiduals(lm2,train_x,train_y)
print(residuals)
print(sum(residuals))
SSR=SSR(lm1,train_x,train_y)
print(sum(SSR))
R=1-(sum(residuals)/sum(SSR))
AdjustedR=1-((1-R)*(5966-1)/(5966-45-1))
y = lm2.predict(train_x)
sns.set(style="whitegrid")
sns.residplot(residuals,y,lowess=True,color="g")
stats.probplot(residuals,dist="norm",plot=pylab)
pylab.show()

pdct2 = lm2.predict(test_x)
print(pdct2)

# mean square error
# -----------------
mse = np.mean((pdct2 - test_y)**2)
print("MSE = {0}, RMSE = {1}".format(mse,math.sqrt(mse)))

actual = list(test_y.head(50))
predicted = np.round(np.array(list(pdct2.head(50))),2)
print(predicted)

df_results = pd.DataFrame({'actual':actual, 'predicted':predicted})
print(df_results)
#########################################################################
import xgboost as xgb
T_train_xgb = xgb.DMatrix(train_x, train_y)

params = {"objective": "reg:linear", "booster":"gblinear"}
gbm = xgb.train(dtrain=T_train_xgb,params=params)

Y_pred = gbm.predict(xgb.DMatrix(pd.DataFrame(test_x)))
residuals = getresiduals(gbm,train_x,train_y)
print(residuals)
print(sum(residuals))
SSR=SSR(lm1,train_x,train_y)
print(sum(SSR))
R=1-(sum(residuals)/sum(SSR))
AdjustedR=1-((1-R)*(5966-1)/(5966-45-1))
mse = np.mean((Y_pred- test_y)**2)
print("MSE = {0}, RMSE = {1}".format(mse,math.sqrt(mse)))
#########################################################################
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(train_x, train_y)  
pred_y = regressor.predict(test_x) 
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pred_y))  
print('Mean Squared Error:', metrics.mean_squared_error(test_y, pred_y))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pred))) 