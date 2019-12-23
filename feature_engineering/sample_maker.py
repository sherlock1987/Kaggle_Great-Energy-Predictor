import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
# del train["meter_reading"]
import time
import gc

path_train = 'C:\\Users\\HP\\Desktop\\Kaggle\\Kaggle\\new_train.csv'
path_test =  'C:\\Users\\HP\\Desktop\\Kaggle\\Kaggle\\new_test.csv'

# df[df['a']>30]
train = pd.read_csv(path_train)

building_num = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
building_num.append(list(range(0,16)))
building_num.append(list(range(109,126)))
building_num.append(list(range(202,218)))
building_num.append(list(range(292,308)))
building_num.append(list(range(600,617)))
building_num.append(list(range(690,706)))
building_num.append(list(range(747,764)))
building_num.append(list(range(789,803)))
building_num.append(list(range(840,857)))
building_num.append(list(range(900,917)))
building_num.append(list(range(1000,1017)))
building_num.append(list(range(1028,1032)))
building_num.append(list(range(1033,1050)))
building_num.append(list(range(1070,1087)))
building_num.append(list(range(1240,1257)))
building_num.append(list(range(1370,1387)))

print(building_num)
column_id = []
for i in range(len(building_num)):
    for j in range(len(building_num[i])):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        column_id.append(building_num[i][j])

print(column_id)
print(len(column_id))

###for hour drop
# column_hour = [0,4,8,12,16,20]
column_hour = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23]

###for day drop
column_day_1  = [2,4,6]
column_day_2  = [1,3,5,7]

train=train[train.building_id.isin(column_id)]
# train=train[train.building_id.isin([1,2])]

train=train[-train.weekday.isin(column_day_1)]
train=train[-train.hour.isin(column_hour)]

train.to_csv('train_tiny_new.csv')
