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

model_data = pd.read_pickle("./model_data.pkl")
model_labels = pd.read_pickle("./model_labels.pkl")

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
scores = cross_val_score(lin_reg, fit_model, model_labels, 
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