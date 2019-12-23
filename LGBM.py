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
import lightgbm.plotting
# 格式化成2016-03-20 11:45:39形式
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

source_path = ""
path_train = source_path+'new_train.csv'
# path_train = 'sample_train_tiny.csv'
path_test =  source_path+'new_test.csv'
path_submission = 'data/original/sample_submission.csv'
train = pd.read_csv(path_train)

print(train.head)
del train['timestamp']

drop_column = ["wind_speed","weekday","floor_count"]
drop_column = ["wind_speed"]

categoricals = ['beaufort_scale',"site_id", "weekday","building_id", "primary_use", "hour", "meter",  "wind_direction","month"]
numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",
              "dew_temperature", 'precip_depth_1_hr',"floor_count"]
print(train.columns)
target = np.log1p(train["meter_reading"])
del train["meter_reading"]
gc.collect()
train = train.drop(drop_column,axis = 1)
feat_cols = categoricals + numericals


folds = 3
seed = 666
kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
models = []
iteration = 0
params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'subsample': 0.8,
            'subsample_freq': 1,
            'learning_rate': 0.05,
            'num_leaves': 1000,
            'feature_fraction': 0.8,
            'lambda_l1': 0.01,
            'lambda_l2': 0.01,
            # 'max_depth' :9,
            'min_child_samples' :26,
            'min_child_weight':0.001,
            'n_estimators':500
}
for train_index, val_index in kf.split(train, train['building_id']):
    iteration += 1
    print('-----'+str(iteration)+'-----')
    train_X = train[feat_cols].iloc[train_index]
    val_X = train[feat_cols].iloc[val_index]
    train_y = target.iloc[train_index]
    val_y = target.iloc[val_index]
    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)
    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)
    gbm = lgb.train(params,
                lgb_train,
                valid_sets=(lgb_train, lgb_eval),
                early_stopping_rounds=100,
                verbose_eval = 100)
    models.append(gbm)
    del train_X, val_X, lgb_train, lgb_eval, train_y, val_y

for i  in range(len(models)):
    name = 'model_500_drop_'+str(i)+'.txt'
    models[i].save_model(name, num_iteration=-1)
    print('complete_saving_process')


test  = pd.read_csv(path_test)
del test['timestamp']
test = test.drop(drop_column,axis = 1)

i=0
res=[]
step_size = 50000
for j in tqdm(range(int(np.ceil(test.shape[0]/50000)))):
    res.append(np.expm1(sum([model.predict(test.iloc[i:i+step_size]) for model in models])/folds))
    i+=step_size

res = np.concatenate(res)
submission = pd.read_csv(path_submission)
submission['meter_reading'] = res
submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# submission.to_csv('submission_'+ str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))+'.csv', index=False)
submission.to_csv("result_2.csv",index = False)

