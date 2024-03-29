###This project is a machine learning model designed to predict max and min temperature for a location
###based on multiple weather teleconnections. Teleconnections are indices that forecasters use to ascertain 
###broad structions in the atmosphere. They are of a semi-permanent nature, with time-scales of days to weeks
###in the case of the NAO, to multiple months in the case of ENSO.
from os import name
from typing import Tuple
from numpy import polynomial
from numpy.ma.core import count, mod
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas.core.dtypes.missing import isna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score



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
Sounding_filepath_1 = "Data/CAM00071867-data.txt"
Sounding_filepath_2 = "Data/CAM00071109-data.txt"
Sounding_filepath_3 = "Data/USM00072206-data.txt" ###There's a lot of missing days. Needs to be QCed
Sounding_filepath_4 = "Data/USM00072381-data.txt"
Sounding_filepath_5 = "Data/USM00072456-data.txt"

Station_data_file_path = "Data/KIAD_Station_data.txt" # Data/KIAD_Station_data.txt  Data/LouisvilleDailyWxSummary.csv Data/KNGU_station_data.txt

Target_Value = "TMAX" #Target_Value is the parameter you want to predict. Can be 'TAVG', 'TMAX', 'TMIN'
######### Reading in Data Files  ##############################
#Data Column is right justified so skipinitialspace must be set to true
observation_start_date = pd.to_datetime("1965-4-1") #for if the observation begins at any other time than the beginning of the dataset
target_forecast = 3
sounding_level = 50000 #in pascals
sounding_data_type = 3 #3 = geopotential height, 4 = temperature, 5 = relative humidity
polynomial_degree = 2 # NTS, best results seem to occur at polynomial_degree = 2

def split_train(data, data_labels, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(shuffled_indices) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[test_indices], data_labels[test_indices], data[train_indices], data_labels[train_indices]

def plot_learning_curves(model, X, y):
    train_data, train_labels, val_data, val_labels = split_train(X, y, test_ratio = 0.2)
    train_errs, val_errs = [], []
    for m in range(1, len(train_data)):
        model.fit(train_data[:m], train_labels[:m])
        label_train_predict = model.predict(train_data[:m])
        label_val_predict = model.predict(val_data)
        train_errs.append(mean_squared_error(label_train_predict, train_labels[:m]))
        val_errs.append(mean_squared_error(label_val_predict, val_labels))
    plt.plot(np.sqrt(train_errs), "r-+", linewidth = 2, label = "Train")
    plt.plot(np.sqrt(val_errs), "b-", linewidth = 3, label = "Validation")
    plt.show

def plot_learning_curves(model, X, y):
    train_data, train_labels, val_data, val_labels = split_train(X, y, test_ratio = 0.2)
    train_errs, val_errs = [], []
    for m in range(1, len(train_data)):
        model.fit(train_data[:m], train_labels[:m])
        label_train_predict = model.predict(train_data[:m])
        label_val_predict = model.predict(val_data)
        train_errs.append(mean_squared_error(label_train_predict, train_labels[:m]))
        val_errs.append(mean_squared_error(label_val_predict, val_labels))
    plt.plot(np.sqrt(train_errs), "r-+", linewidth = 2, label = "Train")
    plt.plot(np.sqrt(val_errs), "b-", linewidth = 3, label = "Validation")
    plt.show()

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

### Function for selecting desired sounding parameters
### Inputs - Filepath: String path to desired sounding location
###          Parameter: integer to select the data desired
###                     Geopotential Height = 3
###                     Temperature =  4
###                     Relative Humidity = 5
###          height: Pressure height of desired parameter
###                     appropriate values are 85000, 70000, 50000, or 20000
### Returns a Pandas series with a Datetime index and parameter data        
def sounding_entry(filepath, parameter, height):
    with open(filepath) as f:
        sounding_lines = f.readlines()
    f.close

    sounding_array = []
    date_index = []
    skipped = False
    desired_sounding = False
    for _, line in enumerate(sounding_lines):
        date_line = line.split()            
                   
        if date_line[0][0] == "#" and date_line[4] == "12":
            if skipped:             ## For error correction. If the previous sounding did not have the associated px level
                date_index.pop()    ## it will discard that date so dates and data remain aligned
            
            date = datetime.datetime(int(date_line[1]), int(date_line[2]), int(date_line[3]))
            date_index.append(date)
            desired_sounding = True
            skipped = True
        elif date_line[0][0] == "#":
            desired_sounding = False
        if desired_sounding:
            line = line.replace("A", " ")
            line = line.replace("B", " ")
            # line = line.replace("\n", " ")
            line = line.split()
            if float(line[2]) == height:
                sounding_array.append(float(line[parameter]))
                skipped = False
        
    date_index = pd.DatetimeIndex(date_index)
    return pd.DataFrame(sounding_array, index= date_index, columns=["Sounding Data"]) #NTS change return to dataframe



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
end_date = SOI_data.index.max() + pd.DateOffset(day=31)
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
#merged_data = merged_data.merge(sounding_entry(Sounding_filepath_1, sounding_data_type, sounding_level), left_index = True, right_index=True)
merged_data = merged_data.merge(sounding_entry(Sounding_filepath_2, sounding_data_type, sounding_level), left_index = True, right_index=True)
# #merged_data = merged_data.merge(sounding_entry(Sounding_filepath_3, sounding_data_type, sounding_level), left_index = True, right_index=True)
#merged_data = merged_data.merge(sounding_entry(Sounding_filepath_4, sounding_data_type, sounding_level), left_index = True, right_index=True)
# #merged_data = merged_data.merge(sounding_entry(Sounding_filepath_5, sounding_data_type, sounding_level), left_index = True, right_index=True)

cleaned_data = merged_data.interpolate()
cleaned_data["MONTH"] = cleaned_data.index.month
cleaned_data[Target_Value +"_forecast"] = cleaned_data.shift(target_forecast)[Target_Value]
# cleaned_data["NAO_rate"] = (cleaned_data.shift(1)["NAO"] - cleaned_data.shift(-1)["NAO"])/2
# cleaned_data["AO_rate"] = (cleaned_data.shift(1)["AO"] - cleaned_data.shift(-1)["AO"])/2
# cleaned_data["PNA_rate"] = (cleaned_data.shift(1)["PNA"] - cleaned_data.shift(-1)["PNA"])/2
# cleaned_data["SOI_rate"] = (cleaned_data.shift(1)["SOI"] - cleaned_data.shift(-1)["SOI"])/2
# test_set, train_set = split_train(cleaned_data, .2)

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


### Adding Polynomial Features to the model ######
poly_features = PolynomialFeatures(degree=polynomial_degree, include_bias= False)
fit_model = poly_features.fit_transform(fit_model)

lin_reg = LinearRegression()

# plot_learning_curves(lin_reg, fit_model_poly, model_labels)
# plot_learning_curves(lin_reg, fit_model_poly, model_labels)

###### cross Validation ####
scores = cross_val_score(lin_reg, 
                         fit_model[target_forecast:], model_labels[target_forecast:], 
                         scoring = "neg_mean_squared_error", cv=10, error_score='raise')

rmse_scores = np.sqrt(-scores)
display_scores(rmse_scores)


# lin_mse = mean_squared_error(model_labels[target_forecast:], model_predictions[target_forecast:])
# lin_rmse = np.sqrt(lin_mse)
# print("MSE: ",lin_mse)
# print("RMSE: ",lin_rmse)









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