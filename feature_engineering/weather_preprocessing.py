import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import datetime
from sklearn.preprocessing import LabelEncoder

# oad Data
weather_train = pd.read_csv('data/original/weather_train.csv')
weather_test = pd.read_csv('data/original/weather_test.csv')
train = pd.read_csv('data/original/train.csv')
test = pd.read_csv('data/original/test.csv')

#Before we merge the dataset into one training dataset, we need to preprocessing the missing value first
#Function that plot missing value statistics
def missing_statistics(df):
    statitics = pd.DataFrame(df.isnull().sum()).reset_index()
    statitics.columns=['COLUMN NAME',"MISSING VALUES"]
    statitics['TOTAL ROWS'] = df.shape[0]
    statitics['% MISSING'] = round((statitics['MISSING VALUES']/statitics['TOTAL ROWS'])*100,2)
    return statitics

#Function that fill missing value with timestamp method
def fill_missing_column(df,filler_df,col):
    null_df = df.loc[df[col].isnull()]

    if null_df.empty != True:
        null_df[col] = null_df.apply(lambda x: filler_df.loc[x['site_id']][x['day']][x['month']], axis=1)
        df.loc[null_df.index, col] = null_df[col]

    return df

def fill_weather_dataset(weather_df):

    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)
    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    missing_hours = []
    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df,new_rows])

        weather_df = weather_df.reset_index(drop=True)

    # Add new Features
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month

    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id','day','month'])

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])
    weather_df.update(air_temperature_filler,overwrite=False)

    # Step 1
    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()
    # Step 2
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])

    weather_df.update(cloud_coverage_filler,overwrite=False)

    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])
    weather_df.update(due_temperature_filler,overwrite=False)

    # Step 1
    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
    # Step 2
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])

    weather_df.update(sea_level_filler,overwrite=False)

    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])
    weather_df.update(wind_direction_filler,overwrite=False)

    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])
    weather_df.update(wind_speed_filler,overwrite=False)

    # Step 1
    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
    # Step 2
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler,overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)
    print(missing_statistics(weather_df))

    return weather_df

weather_train = fill_weather_dataset(weather_train)
weather_test = fill_weather_dataset(weather_test)

def degToCompass(num):
    val=int((num/22.5)+ 0.5)
    arr=[i for i in range(0,16)]
    return arr[(val % 16)]
def weather_preprocessing(data):
    #process time data
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["weekday"] = data["timestamp"].dt.weekday
    data["hour"] = data["timestamp"].dt.hour
    data["month"] = data["timestamp"].dt.month
    data["weekday"] = data['weekday'].astype(np.uint8)
    data["hour"] = data['hour'].astype(np.uint8)
    data["month"] = data["month"].astype(np.uint8)
    # data['year_built'] = data['year_built']-np.min(data['year_built'].values)
    #暂时不做任何的处理
    # data['square_feet'] = np.log(data['square_feet'])
    beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9),
              (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]
    for item in beaufort:
        data.loc[(data['wind_speed']>=item[1]) & (data['wind_speed']<item[2]), 'beaufort_scale'] = item[0]
    # del data["timestamp"]
    
    data['wind_direction'] = data['wind_direction'].apply(degToCompass)
    data['beaufort_scale'] = data['beaufort_scale'].astype(np.uint8)
    data["wind_direction"] = data['wind_direction'].astype(np.uint8)
    data["site_id"] = data['site_id'].astype(np.uint8)
    return data

weather_train = weather_preprocessing(weather_train)
weather_test = weather_preprocessing(weather_test)
