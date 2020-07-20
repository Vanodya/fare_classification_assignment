import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from _datetime import datetime
from math import sin, cos, sqrt, atan2, radians


test = pd.read_csv("E:/MSc/Sem 02/Machine Learning/Kaggle/test.csv")
train = pd.read_csv("E:/MSc/Sem 02/Machine Learning/Kaggle/train.csv")

R = 6373 # Radius of earth in km

# Train
train['true_duration'] = 0
for i in range(0,len(train)):
    train.at[i,'true_duration'] =(datetime.strptime(train['drop_time'][i], '%m/%d/%Y %H:%M') -
                                   datetime.strptime(train['pickup_time'][i], '%m/%d/%Y %H:%M')).seconds

for j in range(0, len(train)):
    train.at[j, 'lat1'] = radians(train['pick_lat'][j])
    train.at[j, 'lon1'] = radians(train['pick_lon'][j])
    train.at[j, 'lat2'] = radians(train['drop_lat'][j])
    train.at[j, 'lon2'] = radians(train['drop_lon'][j])

train['dlon'] = train['lon2'] - train['lon1']
train['dlat'] = train['lat2'] - train['lat1']
train['c'] = 0

for i in range(0, len(train)):
    a = sin(train['dlat'][i] / 2)**2 + cos(train['lat1'][i]) * cos(train['lat2'][i]) * sin(train['dlon'][i] / 2)**2
    train.loc[i,'c'] = 2 * atan2(sqrt(a), sqrt(1 - a))

train['distance'] = R * train['c']
train = train.drop(['tripid', 'pickup_time', 'drop_time', 'pick_lon', 'drop_lat', 'drop_lon', 'lat1', 'lon1', 'lat2', 'lon2',
            'dlon', 'dlat', 'c'], axis = 1)

print("Train data set is done")

# Test
test['true_duration'] = 0
for i in range(0,len(test)):
    test.at[i,'true_duration'] =(datetime.strptime(test['drop_time'][i], '%m/%d/%Y %H:%M') -
                                   datetime.strptime(test['pickup_time'][i], '%m/%d/%Y %H:%M')).seconds

for j in range(0, len(test)):
    test.at[j, 'lat1'] = radians(test['pick_lat'][j])
    test.at[j, 'lon1'] = radians(test['pick_lon'][j])
    test.at[j, 'lat2'] = radians(test['drop_lat'][j])
    test.at[j, 'lon2'] = radians(test['drop_lon'][j])

test['dlon'] = test['lon2'] - test['lon1']
test['dlat'] = test['lat2'] - test['lat1']
test['c'] = 0

for i in range(0, len(test)):
    a = sin(test['dlat'][i] / 2)**2 + cos(test['lat1'][i]) * cos(test['lat2'][i]) * sin(test['dlon'][i] / 2)**2
    test.loc[i,'c'] = 2 * atan2(sqrt(a), sqrt(1 - a))

test['distance'] = R * test['c']
test = test.drop(['tripid', 'pickup_time', 'drop_time', 'pick_lon', 'drop_lat', 'drop_lon', 'lat1', 'lon1', 'lat2', 'lon2',
           'dlon', 'dlat', 'c'], axis = 1)

print("Test data set is done")

params = {'fit_intercept': [True], "class_weight": ["balanced"]}
base_model = LogisticRegression().fit(train.loc[:,test.columns != 'label'], train['label'])
init_model = GridSearchCV(estimator=base_model, param_grid=params)
logistic_model = init_model.fit(train.loc[:,test.columns != 'label'], train['label']).best_estimator_

print("Model building is done")

predictions = pd.DataFrame(logistic_model.predict(test))
predictions.columns = ['predicted_class']

# predictions.to_csv("E:/MSc/Sem 02/Machine Learning/Kaggle/results/logistic_predictions.csv")
