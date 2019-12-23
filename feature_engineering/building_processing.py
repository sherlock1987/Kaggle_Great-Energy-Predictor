import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import datetime

warnings.simplefilter('ignore')
matplotlib.rcParams['figure.dpi'] = 100
sns.set()
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

building = pd.read_csv('data/original/building_metadata.csv')
train = pd.read_csv("data/original/train.csv")

print(train.shape)
print(building.head())
print(building.shape)

def find_relation():
    sns.pairplot(building.drop(columns= 'building_id'), kind="reg",diag_kind = 'kde')
    plt.show()

def floor_count_distribution(data):
    building = data
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set(xlabel='Floor Count', ylabel='#')
    building['floor_count'].value_counts(dropna=False).sort_index().plot(ax=ax)
    ax.legend(['Train', 'Test'])
    num_missing = pd.DataFrame(building.isna().sum().sort_values(ascending=False), columns=['NaN Count'])
    print(num_missing)
    plt.show()

def year_build_distribution():
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set(xlabel='Year Built', ylabel='# Of Buildings', title='Buildings built in each year')
    train['meter_reading'].value_counts(dropna=False).sort_index().plot(ax=ax)
    ax.legend(['Train', 'Test'])
    num_missing = pd.DataFrame(building.isna().sum().sort_values(ascending=False), columns=['NaN Count'])
    print(num_missing)
    plt.show()
def meter_reading_dis():
    print(train['meter_reading'])
    list_meter_reading = list(train['meter_reading'])
    print(np.max(list_meter_reading))
    print(np.min(list_meter_reading))
    print(np.mean(list_meter_reading))

# imputer for building
def imputer(data_training,model):
    building_training = data_training
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    building_np = building_training.values
    imp.fit(building_np[0:,4:6])
    le = LabelEncoder()
    building_np[:,2] = le.fit_transform(building_np[:,2])
    building_np[0:,4:6] = imp.transform(building_np[0:,4:6])
    building_data_after=pd.DataFrame(data = building_np,columns = ["site_id","building_id","primary_use","square_feet","year_built","floor_count"])
    building_data_after.to_csv(path_or_buf= ("building_data_after_imputation_"+model+".csv"),index = False)


