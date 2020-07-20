import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
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
train.drop(['tripid', 'pickup_time', 'drop_time', 'pick_lon', 'drop_lat', 'drop_lon', 'lat1', 'lon1', 'lat2', 'lon2',
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
test.drop(['tripid', 'pickup_time', 'drop_time', 'pick_lon', 'drop_lat', 'drop_lon', 'lat1', 'lon1', 'lat2', 'lon2',
           'dlon', 'dlat', 'c'], axis = 1)

print("Test data set is done")

n_estimators = [int(x) for x in np.linspace(start=200,stop=1000,num=5)]
max_features = ['auto']
max_depth = [int(x) for x in np.linspace(10,50,num=5)]
max_depth.append(None)
min_samples_split = [2]
bootstrap = [True,False]
random_grid = {'n_estimators':n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'bootstrap':bootstrap}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions= random_grid, n_iter= 100, cv = 10, verbose=2,
                               random_state=42,n_jobs=-1)
rf_random.fit(train.loc[:,train.columns != 'label'], train['label'])
predictions = pd.DataFrame(rf_random.predict(test.loc[:,test.columns != 'label']))
predictions.columns = ['predicted_class']

# predictions.to_csv("E:/MSc/Sem 02/Machine Learning/Kaggle/results/predictions.csv")
