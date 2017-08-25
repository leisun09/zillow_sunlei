# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 09:08:27 2017

@author: gzsunlei
"""

#coding=utf-8
# XGBoost baseline for feature engineering.
# 1. Read data.
# 2. Encode missing data.
# 3. Split training data to two parts for training and validation.
# 4. Predict the test data.
# 5. Output to file.
#
# Training result: [189] train-mae:0.066996 valid-mae:0.065312
# Public score: 0.0656603

# 训练集特征和测试集特征分布严重不一致的特征考虑去掉
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import sys
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt

# Parameters
XGB_WEIGHT = 0.6300
BASELINE_WEIGHT = 0.0056
OLS_WEIGHT = 0.0550

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg

def getStructureFeature(train_df):
    #polnomials of the variable
#    train_df["N-structuretaxvaluedollarcnt-2"] = train_df["structuretaxvaluedollarcnt"] ** 2
#    train_df["N-structuretaxvaluedollarcnt-3"] = train_df["structuretaxvaluedollarcnt"] ** 3

    #Average structuretaxvaluedollarcnt by city
    group = train_df.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
    train_df['N-Avg-structuretaxvaluedollarcnt'] = train_df['regionidcity'].map(group)

    #Deviation away from average
#    train_df['N-Dev-structuretaxvaluedollarcnt'] = abs((train_df['structuretaxvaluedollarcnt'] - train_df['N-Avg-structuretaxvaluedollarcnt']))/train_df['N-Avg-structuretaxvaluedollarcnt']

def getLocationFeature(train_df):
    #Number of properties in the zip
    zip_count = train_df['regionidzip'].value_counts().to_dict()
#    train_df['N-zip_count'] = train_df['regionidzip'].map(zip_count)

    #Number of properties in the city
    city_count = train_df['regionidcity'].value_counts().to_dict()
#    train_df['N-city_count'] = train_df['regionidcity'].map(city_count)

    #Number of properties in the city
    region_count = train_df['regionidcounty'].value_counts().to_dict()
#    train_df['N-county_count'] = train_df['regionidcounty'].map(region_count)

def getTaxFeature(train_df):
    #Ratio of tax of property over parcel
#    train_df['N-ValueRatio'] = train_df['taxvaluedollarcnt']/train_df['taxamount']

    #TotalTaxScore
#    train_df['N-TaxScore'] = train_df['taxvaluedollarcnt']*train_df['taxamount']

    #polnomials of tax delinquency year
#    train_df["N-taxdelinquencyyear-2"] = train_df["taxdelinquencyyear"] ** 2
#    train_df["N-taxdelinquencyyear-3"] = train_df["taxdelinquencyyear"] ** 3

    #Length of time since unpaid taxes
#    train_df['N-life'] = 2018 - train_df['taxdelinquencyyear']
    pass

def getPropertyFeature(train_df):
    #life of property
    train_df['N-life'] = 2018 - train_df['yearbuilt']

    #error in calculation of the finished living area of home
    train_df['N-LivingAreaError'] = train_df['calculatedfinishedsquarefeet']/train_df['finishedsquarefeet12']

    #proportion of living area
    train_df['N-LivingAreaProp'] = train_df['calculatedfinishedsquarefeet']/train_df['lotsizesquarefeet']
    train_df['N-LivingAreaProp2'] = train_df['finishedsquarefeet12']/train_df['finishedsquarefeet15']

    #Amout of extra space
    train_df['N-ExtraSpace'] = train_df['lotsizesquarefeet'] - train_df['calculatedfinishedsquarefeet']
    train_df['N-ExtraSpace-2'] = train_df['finishedsquarefeet15'] - train_df['finishedsquarefeet12']

    #Total number of rooms
    train_df['N-TotalRooms'] = train_df['bathroomcnt']*train_df['bedroomcnt']

    #Average room size
    train_df['N-AvRoomSize'] = train_df['calculatedfinishedsquarefeet']/train_df['roomcnt']

    # Number of Extra rooms
    train_df['N-ExtraRooms'] = train_df['roomcnt'] - train_df['N-TotalRooms']

    #Ratio of the built structure value to land area
    train_df['N-ValueProp'] = train_df['structuretaxvaluedollarcnt']/train_df['landtaxvaluedollarcnt']

    #Does property have a garage, pool or hot tub and AC?
    train_df['N-GarPoolAC'] = ((train_df['garagecarcnt']>0) & (train_df['pooltypeid10']>0) & (train_df['airconditioningtypeid']!=5))*1

    train_df["N-location"] = train_df["latitude"] + train_df["longitude"]
    train_df["N-location-2"] = train_df["latitude"]*train_df["longitude"]
    train_df["N-location-2round"] = train_df["N-location-2"].round(-4)

    train_df["N-latitude-round"] = train_df["latitude"].round(-4)
    train_df["N-longitude-round"] = train_df["longitude"].round(-4)

def getDateFeature(train_df): #base-->0.064924
    def seas(x):
        if 1 <= x <= 3:
            return 1        #--- Spring
        elif 4 <= x <= 6:
            return 2        #---Summer
        elif 7 <= x <= 9:
            return 3        #--- Fall (Autumn)
        else:
            return 4        #--- Winter

    if 'transactiondate' in train_df.columns:
        train_df['transactiondate'] = pd.to_datetime(train_df['transactiondate'])
    else:
        train_df['transactiondate'] = pd.to_datetime('20160101')
    #--- Creating three additional columns each for the month and day ---
    train_df['transaction_month'] = train_df.transactiondate.dt.month.astype(np.int64)
#    train_df['transaction_day'] = train_df.transactiondate.dt.weekday.astype(np.int64)
#    train_df['transaction_season'] = train_df['transaction_month'].apply(seas)
    #--- Dropping the 'transactiondate' column now ---
    train_df.drop('transactiondate', axis=1, inplace=True)

def getNewFeature(train_df): #base:0.064992
    getDateFeature(train_df)
    getPropertyFeature(train_df)
    getTaxFeature(train_df)
    getLocationFeature(train_df)
#    getStructureFeature(train_df)

print('Reading training data, properties and test data.')
train = pd.read_csv("../data/train_2016_v2.csv")
properties = pd.read_csv('../data/properties_2016.csv')
test = pd.read_csv('../data/sample_submission.csv')

print( "\nProcessing data for LightGBM ..." )
for c, dtype in zip(properties.columns, properties.dtypes):
    if dtype == np.float64:
        properties[c] = properties[c].astype(np.float32)


print('Encoding missing data.')
for column in properties.columns:
    if properties[column].dtype == 'object':
        properties[column].fillna(-1, inplace=True)
        label_encoder = LabelEncoder()
        list_value = list(properties[column].values)
        label_encoder.fit(list_value)
        properties[column] = label_encoder.transform(list_value)

print('Creating training and validation data for xgboost.')
train_with_properties = train.merge(properties, how='left', on='parcelid')
getNewFeature(train_with_properties)

dropcol = ['parcelid', 'logerror', 'transactiondate'] + ['finishedsquarefeet12', 'taxdelinquencyyear', 'N-LivingAreaProp2', 'N-ExtraSpace-2', 'finishedsquarefeet6', 'latitude', 'N-LivingAreaProp', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet', 'heatingorsystemtypeid', 'buildingqualitytypeid', 'finishedfloor1squarefeet', 'N-ExtraRooms', 'N-TotalRooms', 'garagetotalsqft', 'garagecarcnt', 'regionidcity', 'propertycountylandusecode', 'landtaxvaluedollarcnt', 'yardbuildingsqft17', 'unitcnt', 'N-latitude-round', 'structuretaxvaluedollarcnt', 'N-ValueProp', 'propertyzoningdesc', 'censustractandblock', 'bathroomcnt', 'N-location', 'N-AvRoomSize', 'hashottuborspa', 'fullbathcnt']

candicol = [col for col in train_with_properties.columns if not col in dropcol]

x_train = train_with_properties.drop(dropcol, axis=1)

y_train = train_with_properties['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

x_train = x_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(x_train, label=y_train)

##### RUN LIGHTGBM

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.5      # feature_fraction -- OK, back to .5, but maybe later increase this
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0

print("\nFitting LightGBM model ...")
clf = lgb.train(params, d_train, 430)

print("\nPrepare for LightGBM prediction ...")
print("   Read sample file ...")
sample = pd.read_csv('../data/sample_submission.csv')
print("   ...")
sample['parcelid'] = sample['ParcelId']
print("   Merge with property data ...")
df_test = sample.merge(properties, on='parcelid', how='left')
getNewFeature(df_test)
x_test = df_test[train_columns]

x_test = x_test.values.astype(np.float32, copy=False)

print("\nStart LightGBM prediction ...")
p_test = clf.predict(x_test)

##### PROCESS DATA FOR XGBOOST
train_df = train_with_properties
# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(dropcol, axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

##### RUN XGBOOST

print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,
    'alpha': 0.4,
    'base_score': y_mean,
    'silent': 1
}

x_test = df_test[train_columns]
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 250
print("num_boost_rounds="+str(num_boost_rounds))

# train model
print( "\nTraining XGBoost ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost ...")
xgb_pred1 = model.predict(dtest)

xgb_pred = xgb_pred1

#linear regression
submission = pd.read_csv("../data/sample_submission.csv")
x_train = x_train.fillna(0)
reg = LinearRegression(n_jobs=-1)
reg.fit(x_train, y_train);

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']

########################
########################
##  Combine and Save  ##
########################
########################


##### COMBINE PREDICTIONS

print( "\nCombining XGBoost, LightGBM, and baseline predicitons ..." )
lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)
xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
pred0 = xgb_weight0*xgb_pred + baseline_weight0*BASELINE_PRED + lgb_weight*p_test

x_test.fillna(0,inplace=True)
for c, dtype in zip(x_test.columns, x_test.dtypes):
    print(c, dtype, max(x_test[c]), sep='\t')
    
print( "\nPredicting with OLS and combining with XGB/LGB/baseline predicitons: ..." )
for i in range(len(test_dates)):
    x_test['transactiondate'] = test_dates[i]
    getDateFeature(x_test)
    for c, dtype in zip(x_test.columns, x_test.dtypes):
        print(i, c, dtype, max(x_test[c]), sep='\t')    
    pred = OLS_WEIGHT*reg.predict(x_test) + (1-OLS_WEIGHT)*pred0
    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
    print('predict...', i)

print( "\nCombined XGB/LGB/baseline/OLS predictions:" )
print( submission.head() )



##### WRITE THE RESULTS

from datetime import datetime

print( "\nWriting results to disk ..." )
submission.to_csv('../data/sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")