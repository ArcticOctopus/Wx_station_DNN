###This project is a machine learning model designed to predict max and min temperature for a location
###based on multiple weather teleconnections. Teleconnections are indices that forecasters use to ascertain 
###broad structions in the atmosphere. They are of a semi-permanent nature, with time-scales of days to weeks
###in the case of the NAO, to multiple months in the case of ENSO.
from typing import Tuple
from numpy.ma.core import count, mod
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas.core.dtypes.missing import isna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# def find_nas(data_set):
#     missing_data = np.where(np.isnan(merged_data))
#     index = mer


# class DataFrameSelector(BaseEstimator,TransformerMixin):
#     def __Init__(self, attribute_names):
#         self.attribute_names = attribute_names
#     def fit(self, X, y = None):
#         return self
#     def transform(self, X):
#         return X[self.attribute_names].values


NAO_file_path = "Data/nao.reanalysis.t10trunc.1948-present.txt"
PNA_file_path = "Data/pna.reanalysis.t10trunc.1948-present.txt"
SOI_file_path = "Data/SOIdata"
AO_file_path = "Data/AOdata"

Station_data_file_path = "Data/KIAD_Station_data.txt" # Data/KIAD_Station_data.txt  Data/LouisvilleDailyWxSummary.csv

Target_Value = "TMAX" #Target_Value is the parameter you want to predict. Can be 'TAVG', 'TMAX', 'TMIN'
######### Reading in Data Files  ##############################
#Data Column is right justified so skipinitialspace must be set to true
observation_start_date = pd.to_datetime("1965-1-1") #for if the observation begins at any other time than the beginning of the dataset
target_forecast = 14

def split_train(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(shuffled_indices) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[test_indices], data.iloc[train_indices]

raw_NAO_data = pd.read_csv(NAO_file_path, 
                           names = ("Year", "Month", "Day", "Data"), 
                           sep = " ", 
                           dtype = {"Data":np.float64}, 
                           skipinitialspace = True)

raw_PNA_data = pd.read_csv(PNA_file_path, 
                           names = ("Year", "Month", "Day", "Data"), 
                           sep = " ", 
                           dtype = {"Data":np.float64}, 
                           skipinitialspace = True)

raw_SOI_data = pd.read_csv(SOI_file_path, 
                           names = ("Date", "Data"), 
                           sep = ",", 
                           dtype = {"Data":np.float64})

raw_AO_data = pd.read_csv(AO_file_path, 
                           names = ("Date", "Data"), 
                           sep = ",", 
                           dtype = {"Data":np.float64})

raw_Station_data = pd.read_csv(Station_data_file_path)


Station_dates= pd.to_datetime(raw_Station_data["DATE"])



Station_data = pd.DataFrame(raw_Station_data[[Target_Value]].values,  columns = [Target_Value], index = Station_dates)

#print(Station_data.head)
#Station_dates = pd.DatetimeIndex(pd.to_datetime())

NAO_dates = pd.DatetimeIndex(pd.to_datetime(raw_NAO_data[["Year", "Month", "Day"]]))

NAO_data =pd.DataFrame(raw_NAO_data[["Data"]].values, columns = ["NAO"], index = NAO_dates)

PNA_dates = pd.DatetimeIndex(pd.to_datetime(raw_PNA_data[["Year", "Month", "Day"]]))

PNA_data =pd.DataFrame(raw_PNA_data[["Data"]].values, columns = ["PNA"], index = PNA_dates)

SOI_dates = pd.to_datetime(raw_SOI_data["Date"].values.flatten(), format = '%Y%m')

SOI_data = pd.DataFrame(raw_SOI_data[["Data"]].values, columns = ['SOI'], index = SOI_dates)


start_date = SOI_data.index.min()- pd.DateOffset(day=1)
end_date = SOI_data.index.max() +pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq = 'D')

SOI_data = SOI_data.reindex(dates, method = 'ffill')
#print(SOI_data.head)

AO_dates = pd.to_datetime(raw_AO_data["Date"].values.flatten(), format = '%Y%m')

AO_data = pd.DataFrame(raw_AO_data[["Data"]].values, columns = ['AO'], index = AO_dates)

start_date = AO_data.index.min()- pd.DateOffset(day=1)
end_date = AO_data.index.max() +pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq = 'D')

AO_data = AO_data.reindex(dates, method = 'ffill')


merged_data = PNA_data.merge(NAO_data, left_index = True, right_index = True)

merged_data = merged_data.merge(SOI_data, left_index = True, right_index = True)
merged_data = merged_data.merge(AO_data, left_index = True, right_index = True)
merged_data = merged_data.merge(Station_data, left_index = True, right_index=True)#, how = "left")

cleaned_data = merged_data.interpolate()
cleaned_data["MONTH"] = cleaned_data.index.month
cleaned_data[Target_Value +"_forecast"] = cleaned_data.shift(target_forecast)[Target_Value]
cleaned_data["NAO_rate"] = (cleaned_data.shift(1)["NAO"] - cleaned_data.shift(-1)["NAO"])/2
cleaned_data["AO_rate"] = (cleaned_data.shift(1)["AO"] - cleaned_data.shift(-1)["AO"])/2
cleaned_data["PNA_rate"] = (cleaned_data.shift(1)["PNA"] - cleaned_data.shift(-1)["PNA"])/2
cleaned_data["SOI_rate"] = (cleaned_data.shift(1)["SOI"] - cleaned_data.shift(-1)["SOI"])/2
test_set, train_set = split_train(cleaned_data, .2)

cleaned_data.drop(index = cleaned_data.index[:target_forecast],axis = 0, inplace=True)
cleaned_data.drop(index = cleaned_data.index[-1],axis = 0, inplace=True)


model_labels = cleaned_data[Target_Value+"_forecast"].copy()
#print(model_labels[-10:])
model_data = cleaned_data.copy()
model_data  = model_data.drop(Target_Value+"_forecast", axis=1) #for when predicting the current day temps based only on teleconnections


predictor_names = list(model_data)

model_pipeline = Pipeline([
                            #('selector', DataFrameSelector[predictor_names]),
                            ('imputer', SimpleImputer(strategy="median")),
                            ('std scaler', StandardScaler())
])

fit_model = model_pipeline.fit_transform(model_data)

lin_reg = LinearRegression()
lin_reg.fit(model_data[target_forecast:], model_labels[target_forecast:])

model_predictions = lin_reg.predict(model_data)
lin_mse = mean_squared_error(model_labels[target_forecast:], model_predictions[target_forecast:])
lin_rmse = np.sqrt(lin_mse)
print("MSE: ",lin_mse)
print("RMSE: ",lin_rmse)
corr_matrix = cleaned_data.loc[observation_start_date:,:].corr()
print(corr_matrix[Target_Value+"_forecast"])








#def vectorize(raw_data):
#    result = np.zeros([len(raw_data[:]),2])
#    for i in range(len(raw_data[:])):
#        result[i,0] = int(str(raw_data["Year"][i]) + str(raw_data["Month"][i]) + str(raw_data["Day"][i]))
#        result[i,1] = raw_data["Data"][i]

#    return result
#for i in range(len(raw_NAO_data[:])):
    
#    NAO_date.append(datetime.date(int(raw_NAO_data[i,0]), int(raw_NAO_data[i,1]), int(raw_NAO_data[i,2])))
#    NAO_data.append(raw_NAO_data[i,3])
#NAO_data = vectorize(raw_NAO_data)
#print(NAO_data[0])

#a = datetime.date.today()
#a = int(a.strftime('%y%m%d'))
#print(a)