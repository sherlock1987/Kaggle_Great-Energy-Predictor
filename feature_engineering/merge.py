import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import datetime
import building_processing
import weather_preprocessing

path_building = 'data/building_data_after_imputation_train.csv'
path_weather_train = 'data/original/weather_train.csv'
path_train = 'data/original/train.csv'
path_test = 'data/original/test.csv'
path_submission = 'data/original/sample_submission.csv'
path_weather_test  = 'data/original/weather_test.csv'


building_df = pd.read_csv(path_building)
weather_train = pd.read_csv(path_weather_train)
train = pd.read_csv(path_train)
#impute weather train
weather_train = weather_preprocessing.fill_weather_dataset(weather_train)
weather_train = weather_preprocessing.weather_preprocessing(weather_train)
print(weather_train)
#impute weather test

#merge for train data
train = train.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")
train["timestamp"] = pd.to_datetime(train["timestamp"])
train = train.merge(weather_train, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"])
del weather_train
del building_df
train.to_csv('new_train.csv', index=False)
print(train.head())
del train

drop_cols = ["sea_level_pressure", "wind_speed"]

##处理test数据
test = pd.read_csv(path_test)
building_df = pd.read_csv(path_building)
test = test.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")
del building_df
test["timestamp"] = pd.to_datetime(test["timestamp"])
weather_test = pd.read_csv(path_weather_test)
weather_test = weather_preprocessing.fill_weather_dataset(weather_test)
weather_test = weather_preprocessing.weather_preprocessing(weather_test)
test = test.merge(weather_test, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")
del weather_test
print(test.head())
test.to_csv('new_test.csv',index = False)
del test

#save file
