import itertools
import gc

import numpy as np
import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

path_data = ""
path_train = path_data + "sample_train_tiny.csv "
# path_train = path_data + "train_sample_test.csv"

RMSE = 0
RMSE_minus = []
minus = 0
#no 12, what happened.
for site in [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15]:
# for site in [0]:
    # site = 0
    np.random.seed(666)
    ## reading train and test data
    df_train = pd.read_csv(path_train)
    df_train = df_train.drop(['month','hour','weekday'],axis = 1)
    # def df_train
    print(df_train.columns)
    print("del complete")
    # def df_train['wind_speed']

    df_train = df_train[df_train.site_id == site]
    df_train.reset_index(drop=True, inplace=True)
    # print(df_train.columns)
    df_train["ds"] = pd.to_datetime(df_train.timestamp, format='%Y-%m-%d %H:%M:%S')
    df_train["y"] = np.log1p(df_train.meter_reading)
    gc.collect()
    print("training loading completed")

    df_valid = df_train.sample(frac = 0.2,replace = True,axis = 0)
    print("validating loading completed")

    exogenous_features = ["square_feet", "year_built", "air_temperature", "cloud_coverage", "dew_temperature", 'precip_depth_1_hr', 'floor_count']
    df_preds = []

    for building in df_train.building_id.unique():
        print(building)
        df_train_building = df_train[df_train.building_id == building]
        df_test_building =  df_valid[df_valid.building_id == building]

        for meter in df_train_building.meter.unique():

            df_train_building_meter = df_train_building[df_train_building.meter == meter]
            df_test_building_meter = df_test_building[df_test_building.meter == meter]

            # df_train_building_meter.dropna(axis=1, how="all", inplace=True)

            print("Building Prophet model for building", building, "and meter", meter)
            ## initializing model
            model_prophet = Prophet()
            remove_features = []
            for feature in exogenous_features:
                if feature in df_train_building_meter.columns:
                    model_prophet.add_regressor(feature)
                else:
                    remove_features.append(feature)
            for feature in remove_features:
                exogenous_features.remove(feature)

            ## building model
            model_prophet.fit(df_train_building_meter[["ds", "y"] + exogenous_features])

            ## forecasting predictions
            forecast = model_prophet.predict(df_test_building_meter[["ds"] + exogenous_features])
            pred = np.expm1(forecast.yhat.values)
            # df_pred = pd.DataFrame({"row_id": df_test_building_meter.row_id.values, "meter_reading": np.expm1(forecast.yhat.values)})
            correct = df_test_building_meter['meter_reading'].values
            minus = pred - correct
            RMSE_minus.append(list(minus))
            # print(RMSE_minus)
            print("Prophet model completed for building", building, "and meter", meter, "\n")
print(RMSE_minus)
RMSE_final = []
for i in range(len(RMSE_minus)):
    print(RMSE_minus[i])
    for j in range(len(RMSE_minus[i])):
        RMSE_final.append(RMSE_minus[i][j])

rmse = np.sqrt(np.mean(np.array(RMSE_final) ** 2))
print(rmse)
# print(rmse)
# RMSE += rmse

# model_prophet.plot_components(forecast)
