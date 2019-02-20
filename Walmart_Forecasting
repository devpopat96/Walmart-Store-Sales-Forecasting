#Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt

#DataSets
feature = pd.read_csv("./features.csv.zip", parse_dates=True)
stores = pd.read_csv("./stores.csv")
train = pd.read_csv("./train.csv.zip", parse_dates=True)
test = pd.read_csv("./test.csv.zip", parse_dates=True)
sample = pd.read_csv("./sampleSubmission.csv.zip")

#Data Cleaning
for store_num, df in feature.groupby(["Store"]):
    df = df.fillna(df.mean())
    feature = feature.fillna(df)
train_all = pd.merge(pd.merge(train, stores), feature)
test_all = pd.merge(pd.merge(test, stores), feature)
train_all['Split'] = 'Train'
test_all['Split'] = 'Test'


X_all = pd.concat([train_all, test_all], sort=False, ignore_index=False)
X_all['Temperature'] = (X_all['Temperature'] - 32) * 5/9
train_all.Date = pd.to_datetime(train_all.Date)
train_all["Year"] = train_all.Date.dt.year
train_all["Week"] = train_all.Date.dt.week
test_all.Date = pd.to_datetime(test_all.Date)
test_all["Year"] = test_all.Date.dt.year
test_all["Week"] = test_all.Date.dt.week
X_all.Date = pd.to_datetime(X_all.Date)
X_all["Year"] = X_all.Date.dt.year
X_all["Week"] = X_all.Date.dt.week
X_all["BeforeChristmas"] = X_all.Week == 51
X_all["InThanksGiving"] = X_all.Week == 47
X_all.drop(["Date"], axis=1, inplace=True)

X_dummied = pd.get_dummies(X_all, columns=["IsHoliday", "Type", "BeforeChristmas", "InThanksGiving"])
X_dummied = pd.get_dummies(X_all, columns=["Type"])

train_all_sorted = train_all.sort_values(["Store", "Dept", "Year", "Week"])
test_all_sorted = test_all.sort_values(["Store", "Dept", "Year", "Week"])
X_dummied_sorted = X_dummied.sort_values(["Store", "Dept", "Year", "Week"])

train_prosecced = X_dummied_sorted[X_dummied_sorted.Split == "Train"].drop(["Split"], axis=1)
test_prosecced = X_dummied_sorted[X_dummied_sorted.Split == "Test"].drop(["Split"], axis=1)

train_X = train_prosecced.drop(["Weekly_Sales"], axis=1)
train_y = train_prosecced.Weekly_Sales
test_X = test_prosecced.drop(["Weekly_Sales"], axis=1)


#Modelling
rfr = RandomForestRegressor(n_estimators=100, verbose=0, n_jobs=30)
rfr.fit(train_X, train_y)


rfr.score(train_X, train_y)

#Predicting

pred = rfr.predict(test_X)
sample.Weekly_Sales = pred
sample.to_csv("./rfr.csv", index=False)
