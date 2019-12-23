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
import time
import gc
import lightgbm.plotting
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
folds = 3

source_path = "data/new_data/"
path_train = source_path+'new_train.csv'
path_test =  source_path+'new_test.csv'
path_submission = 'data/original/sample_submission.csv'
drop_column = ["wind_speed"]

test  = pd.read_csv(path_test)
del test['timestamp']
test = test.drop(drop_column,axis = 1)

models = []
model_1 = lgb.Booster(model_file='modules/model_500_drop_0.txt')
model_2 = lgb.Booster(model_file='modules/model_500_drop_1.txt')
model_3 = lgb.Booster(model_file='modules/model_500_drop_2.txt')
models.append(model_1)
models.append(model_2)
models.append(model_3)

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
submission.to_csv("output.csv",index = False)

