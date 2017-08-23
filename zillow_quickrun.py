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


    train_df['transactiondate'] = pd.to_datetime(train_df['transactiondate'])

    #--- Creating three additional columns each for the month and day ---
    train_df['transaction_month'] = train_df.transactiondate.dt.month.astype(np.int64)
#    train_df['transaction_day'] = train_df.transactiondate.dt.weekday.astype(np.int64)
#    train_df['transaction_season'] = train_df['transaction_month'].apply(seas)
    #--- Dropping the 'transactiondate' column now ---
    train_df = train_df.drop('transactiondate', 1)

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

train_index = []
valid_index = []
for i in range(len(train_with_properties)):
    if i % 10 != 0:
        train_index.append(i)
    else:
        valid_index.append(i)

train_dataset = train_with_properties.iloc[train_index]
valid_dataset = train_with_properties.iloc[valid_index]

x_train = train_dataset.drop(dropcol , axis=1)
y_train = train_dataset['logerror'].values
x_valid = valid_dataset.drop(dropcol, axis=1)
y_valid = valid_dataset['logerror'].values

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

print('Training the model.')
# xgboost params
params = {
    'eta': 0.033,
    'max_depth': 4,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1
}

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
model = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=100,
                  verbose_eval=1000)

