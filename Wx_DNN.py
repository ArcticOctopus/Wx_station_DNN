###This project is a machine learning model designed to predict max and min temperature for a location
###based on multiple weather teleconnections. Teleconnections are indices that forecasters use to ascertain 
###broad structions in the atmosphere. They are of a semi-permanent nature, with time-scales of days to weeks
###in the case of the NAO, to multiple months in the case of ENSO.
from os import name
from pickle import FALSE
from typing import Tuple
from keras import layers

from numpy import concatenate, polynomial
from numpy.ma.core import count, mod
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import keras
import sklearn
from sklearn.model_selection import train_test_split
from pandas.core.dtypes.missing import isna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import tensorflow as tf
#from tensorflow.python.util.tf_export import KERAS_API_NAME
from sklearn.model_selection import cross_val_score
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.decomposition import PCA



do_Random_CV_Initiation = False

do_Grid_CV_Initiation = False

do_Standard_Initiation = False

do_wide_and_deep = False

plot_history = False

do_Single_layer = False

do_three_model = True

do_seven_model = False

do_fourteen_model = False

do_thirty_model = False


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
model_pipeline = Pipeline([
                            #('selector', DataFrameSelector[predictor_names]),
                            ('imputer', SimpleImputer(strategy="median")),
                            ('std scaler', StandardScaler()),
                            #('PCA', PCA(n_components=.95))                             #####<<<<<<<<<< Adjust if you want PMC
])
rnd_param_distribs = {
                    "n_hidden": [1,3,5,7,9],
                    "n_neurons": [100,250,500,750,1000],
                    "learning_rate": reciprocal(3e-4, 3e-2),
                    "dr_rate": [0.0, 0.1, 0.2, 0.3, 0.4]
                    }

grid_param_distribs = {
                    "n_hidden": [2,3,4],
                    "n_neurons": [600,650,700,750,800],
                    "learning_rate": [0.00066],
                    "dr_rate": [ 0.15, 0.2, 0.25, ]
                    }
### Function to split data into a training set and a test set. Ratio of the split is 
### set by the user. Returns: test_data, test_labels, train_data, train_labels
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

def build_model(n_hidden = 1, n_neurons= 32, learning_rate = 3e-1, input_shape = [150], batch_size = 32, dr_rate = 0):
    model = keras.models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(layers.Dense(n_neurons, activation="selu", kernel_initializer= "lecun_normal"))
        model.add(layers.Dropout(rate=dr_rate))  
    model.add(layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss = "huber", optimizer=optimizer)
    return model

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())


train_data = pd.read_pickle("./train_data.pkl")
train_labels = pd.read_pickle("./train_labels.pkl")
test_data = pd.read_pickle("./test_data.pkl")
test_labels = pd.read_pickle("./test_labels.pkl")
print(train_data.shape)
predictor_names = list(train_data)



fit_train = model_pipeline.fit_transform(train_data)
#model_pipeline.set_params(PCA__n_components = len(fit_train[0,:]))
fit_test = model_pipeline.fit_transform(test_data)
#test_data, test_labels, train_data, train_labels = split_train(fit_model, train_labels, .1)

print(fit_train.shape)
print(fit_test.shape)
### Creating Training, Validation and Test sets ######

#X_train, X_test, Y_train, Y_test = train_test_split(fit_model, train_labels, test_size = .2, random_state= 42)
#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = .3, random_state=42)

#### Model Initiation ######

##  Random CV Initiation    ##

if do_Random_CV_Initiation:
    keras_reg =  tf.keras.wrappers.scikit_learn.KerasRegressor(build_model)
    rnd_search_cv = RandomizedSearchCV(keras_reg, rnd_param_distribs, n_iter= 10, cv = 3)
    rnd_search_cv.fit(fit_train, 
                train_labels, 
                validation_split = .2, 
                epochs = 100, 
                batch_size = 32, 
                callbacks=[keras.callbacks.EarlyStopping(patience=4)])

    print(rnd_search_cv.best_params_)
    print(rnd_search_cv.best_score_)


if do_Grid_CV_Initiation:
    keras_reg =  tf.keras.wrappers.scikit_learn.KerasRegressor(build_model)
    grid_search_cv = GridSearchCV(keras_reg, grid_param_distribs, cv = 3)
    grid_search_cv.fit(fit_train, 
                train_labels, 
                validation_split = .2, 
                epochs = 100, 
                batch_size = 32, 
                callbacks=[keras.callbacks.EarlyStopping(patience=4)])

    print(grid_search_cv.best_params_)
    print(grid_search_cv.best_score_)

##  Standard Initiation ##

if do_Standard_Initiation:
    # print("Total Dataset size ", len(train_data))
    input_A = layers.Input(shape = fit_train.shape[1:], name = "Main Input")
    hidden_1 = layers.Dense(700, activation="selu", kernel_initializer= "lecun_normal")(input_A)
    dropout_2 = layers.Dropout(rate=0.15)(hidden_1)
    hidden_2 = layers.Dense(700, activation="selu", kernel_initializer= "lecun_normal")(dropout_2)
    dropout_3 = layers.Dropout(rate=0.15)(hidden_2)
    hidden_3 = layers.Dense(700, activation="selu", kernel_initializer= "lecun_normal")(dropout_3)
    dropout_4 = layers.Dropout(rate=0.15)(hidden_3)
    hidden_4 = layers.Dense(700, activation="selu", kernel_initializer= "lecun_normal")(dropout_4)
    dropout_5 = layers.Dropout(rate=0.15)(hidden_4)
    hidden_5 = layers.Dense(700)(dropout_5)
    output = layers.Dense(1)(hidden_4)  
    model = keras.models.Model(inputs = [input_A], outputs = [output] )


    model.compile(loss = "huber",
                optimizer = tf.keras.optimizers.SGD(learning_rate=0.0015)
                )
    history = model.fit(fit_train, train_labels, 
                        validation_split = .2, 
                        epochs = 40, 
                        batch_size=32,
                        callbacks=[keras.callbacks.EarlyStopping(patience=10)])      
    forecast_values = model.predict(fit_test)
    mse_test = model.evaluate(fit_test, test_labels)
    print(mse_test)
    plt.scatter(forecast_values, test_labels)
    plt.plot((30,100),(30,100))   
    plt.xlabel("Predicted Max Temp")
    plt.ylabel("Actual Max Temp")  
    plt.show() 

if do_three_model:
    # print("Total Dataset size ", len(train_data))
    input_A = layers.Input(shape = fit_train.shape[1:], name = "Main Input")
    hidden_1 = layers.Dense(350, activation="selu", kernel_initializer= "lecun_normal")(input_A)
    dropout_2 = layers.Dropout(rate=0.1)(hidden_1)
    hidden_2 = layers.Dense(350, activation="selu", kernel_initializer= "lecun_normal")(dropout_2)
    output = layers.Dense(1)(hidden_2)  
    model = keras.models.Model(inputs = [input_A], outputs = [output] )


    model.compile(loss = "huber",
                optimizer = tf.keras.optimizers.SGD(learning_rate=0.008)
                )
    history = model.fit(fit_train, train_labels, 
                        validation_split = .2, 
                        epochs = 40, 
                        batch_size=32,
                        callbacks=[keras.callbacks.EarlyStopping(patience=5)])      
    forecast_values = model.predict(fit_test)
    mse_test = model.evaluate(fit_test, test_labels)
    print(mse_test)
    plt.scatter(forecast_values, test_labels)
    plt.plot((30,100),(30,100))   
    plt.xlabel("Predicted Max Temp")
    plt.ylabel("Actual Max Temp")  
    plt.show() 

if do_seven_model:
    # print("Total Dataset size ", len(train_data))
    input_A = layers.Input(shape = fit_train.shape[1:], name = "Main Input")
    hidden_1 = layers.Dense(350, activation="elu", kernel_initializer= "lecun_normal")(input_A)
    dropout_2 = layers.Dropout(rate=0.1)(hidden_1)
    hidden_2 = layers.Dense(350, activation="elu", kernel_initializer= "lecun_normal")(dropout_2)
    dropout_3 = layers.Dropout(rate=0.1)(hidden_2)
    hidden_3 = layers.Dense(350, activation="elu", kernel_initializer= "lecun_normal")(dropout_3)
    dropout_4 = layers.Dropout(rate=0.1)(hidden_3)
    hidden_4 = layers.Dense(350, activation="elu", kernel_initializer= "lecun_normal")(dropout_4)
    dropout_5 = layers.Dropout(rate=0.1)(hidden_4)
    hidden_5 = layers.Dense(350, activation="elu", kernel_initializer= "lecun_normal")(dropout_5)
    dropout_6 = layers.Dropout(rate=0.1)(hidden_5)
    hidden_6 = layers.Dense(350, activation="elu", kernel_initializer= "lecun_normal")(dropout_6)
    output = layers.Dense(1)(hidden_6)  
    model = keras.models.Model(inputs = [input_A], outputs = [output] )


    model.compile(loss = "huber",
                optimizer = tf.keras.optimizers.SGD(learning_rate=0.0002)
                )
    history = model.fit(fit_train, train_labels, 
                        validation_split = .2, 
                        epochs = 100, 
                        batch_size=32,
                        callbacks=[keras.callbacks.EarlyStopping(patience=12)])      
    forecast_values = model.predict(fit_test)
    mse_test = model.evaluate(fit_test, test_labels)
    print(mse_test)
    plt.scatter(forecast_values, test_labels)
    plt.plot((30,100),(30,100))   
    plt.xlabel("Predicted Max Temp")
    plt.ylabel("Actual Max Temp")  
    plt.show()

if do_fourteen_model:
    # print("Total Dataset size ", len(train_data))
    input_A = layers.Input(shape = fit_train.shape[1:], name = "Main Input")
    hidden_1 = layers.Dense(350, activation="elu", kernel_initializer= "lecun_normal")(input_A)
    dropout_2 = layers.Dropout(rate=0.15)(hidden_1)
    hidden_2 = layers.Dense(350, activation="elu", kernel_initializer= "lecun_normal")(dropout_2)
    dropout_3 = layers.Dropout(rate=0.15)(hidden_2)
    hidden_3 = layers.Dense(350, activation="elu", kernel_initializer= "lecun_normal")(dropout_3)
    dropout_4 = layers.Dropout(rate=0.15)(hidden_3)
    hidden_4 = layers.Dense(350, activation="elu", kernel_initializer= "lecun_normal")(dropout_4)
    dropout_5 = layers.Dropout(rate=0.15)(hidden_4)
    hidden_5 = layers.Dense(350, activation="elu", kernel_initializer= "lecun_normal")(dropout_5)
    output = layers.Dense(1)(hidden_5)  
    model = keras.models.Model(inputs = [input_A], outputs = [output] )


    model.compile(loss = "huber",
                optimizer = tf.keras.optimizers.SGD(learning_rate=0.0002)
                )
    history = model.fit(fit_train, train_labels, 
                        validation_split = .2, 
                        epochs = 100, 
                        batch_size=32,
                        callbacks=[keras.callbacks.EarlyStopping(patience=12)])      
    forecast_values = model.predict(fit_test)
    mse_test = model.evaluate(fit_test, test_labels)
    print(mse_test)
    plt.scatter(forecast_values, test_labels)
    plt.plot((30,100),(30,100))   
    plt.xlabel("Predicted Max Temp")
    plt.ylabel("Actual Max Temp")  
    plt.show()
if do_thirty_model:
    # print("Total Dataset size ", len(train_data))
    input_A = layers.Input(shape = fit_train.shape[1:], name = "Main Input")
    hidden_1 = layers.Dense(7000, activation="elu", kernel_initializer= "lecun_normal")(input_A)
    dropout_2 = layers.Dropout(rate=0.2)(hidden_1)
    hidden_2 = layers.Dense(700, activation="elu", kernel_initializer= "lecun_normal")(dropout_2)
    dropout_3 = layers.Dropout(rate=0.1)(hidden_2)
    hidden_3 = layers.Dense(700, activation="elu", kernel_initializer= "lecun_normal")(dropout_3)
    dropout_4 = layers.Dropout(rate=0.1)(hidden_3)
    hidden_4 = layers.Dense(700, activation="elu", kernel_initializer= "lecun_normal")(dropout_4)
    dropout_5 = layers.Dropout(rate=0.1)(hidden_4)
    hidden_5 = layers.Dense(700, activation="elu", kernel_initializer= "lecun_normal")(dropout_5)
    dropout_6 = layers.Dropout(rate=0.1)(hidden_5)
    hidden_6 = layers.Dense(700, activation="elu", kernel_initializer= "lecun_normal")(dropout_6)
    dropout_7 = layers.Dropout(rate = 0.1)(hidden_6)
    hidden_7 = layers.Dense(700, activation="elu", kernel_initializer= "lecun_normal")(dropout_5)
    dropout_8 = layers.Dropout(rate=0.1)(hidden_5)
    hidden_8 = layers.Dense(700, activation="elu", kernel_initializer= "lecun_normal")(dropout_6)
    output = layers.Dense(1)(hidden_8)  
    
    model = keras.models.Model(inputs = [input_A], outputs = [output] )


    model.compile(loss = "huber",
                optimizer = tf.keras.optimizers.SGD(learning_rate=0.00066)
                )
    history = model.fit(fit_train, train_labels, 
                        validation_split = .2, 
                        epochs = 40, 
                        batch_size=32,
                        callbacks=[keras.callbacks.EarlyStopping(patience=6)])      
    forecast_values = model.predict(fit_test)
    mse_test = model.evaluate(fit_test, test_labels)
    print(mse_test)
    plt.scatter(forecast_values, test_labels)
    plt.plot((30,100),(30,100))   
    plt.xlabel("Predicted Max Temp")
    plt.ylabel("Actual Max Temp")  
    plt.show() 



if do_Single_layer:
    # print("Total Dataset size ", len(train_data))
    input_A = layers.Input(shape = fit_train.shape[1:], name = "Main Input")
    hidden_1 = layers.Dense(750, activation="relu")(input_A)
    
    output = layers.Dense(1)(hidden_1)  
    model = keras.models.Model(inputs = [input_A], outputs = [output] )


    model.compile(loss = "huber",
                optimizer = tf.keras.optimizers.SGD(learning_rate=0.015)
                )
    history = model.fit(fit_train, train_labels, 
                        validation_split = .2, 
                        epochs = 40, 
                        batch_size=100,
                        callbacks=[keras.callbacks.EarlyStopping(patience=10)])      
    forecast_values = model.predict(fit_test)
    plt.scatter(forecast_values, test_labels)
    plt.plot((30,100),(30,100))   
    plt.xlabel("Predicted Max Temp")
    plt.ylabel("Actual Max Temp")  
    plt.show() 
if do_wide_and_deep:
    # print("Total Dataset size ", len(train_data))
    input_A = layers.Input(shape = fit_train.shape[1:], name = "Main Input")
    #dropout_1 = layers.Dropout(rate=0.2)(input_A)
    input_B = layers.Input(shape = [4], name = "Teleconnections Input")
    input_C = layers.Input(shape = [1], name = "Month Input")
    hidden_1 = layers.Dense(500, activation="relu")(input_A)
    #dropout_2 = layers.Dropout(rate=0.2)(hidden_1)
    hidden_2 = layers.Dense(500, activation="relu")(hidden_1)
    #dropout_3 = layers.Dropout(rate=0.2)(hidden_2)
    hidden_3 = layers.Dense(500, activation="relu")(hidden_2)
    #dropout_4 = layers.Dropout(rate=0.2)(hidden_3)
    concat = layers.concatenate([input_B, input_C, hidden_3])
    hidden_4 = layers.Dense(500, activation = 'relu')(hidden_3)
    output = layers.Dense(1)(hidden_4)  
    model = keras.models.Model(inputs = [input_A, input_B, input_C], outputs = [output] )

    model.compile(loss = "huber",
                optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
                )
    
    history = model.fit((fit_train, fit_train[:, :4], fit_train[:, -1]), train_labels, 
                        validation_split = .2, 
                        epochs = 40, 
                        batch_size=32,
                        callbacks=[keras.callbacks.EarlyStopping(patience=4)])    
    mse_test = model.evaluate((fit_test, fit_test[:, :4], fit_test[:, -1]), test_labels)
    print(mse_test)
if plot_history:
    pd.DataFrame(history.history).plot(figsize = (8,5))
    plt.grid(True)
    #plt.gca().set_ylim(0,1)
    plt.show()



# #model.summary()

# # print("Train size ",len(X_train), len(Y_train))
# # print("Train size ",len(X_test), len(Y_test))

# # scores = cross_val_score(lin_reg, 
# #                          fit_model[target_forecast:], train_labels[target_forecast:], 
# #                          scoring = "neg_mean_squared_error", cv=10, error_score='raise')

# # rmse_scores = np.sqrt(-scores)
# # display_scores(rmse_scores)

# history = model.fit(fit_model, train_labels, validation_split = .2, epochs = 40, batch_size=32)
# # mse_test = model.evaluate(X_test, Y_test)
# # X_new = X_test[:3]
# # y_pred = model.predict(X_new)

# # model.evaluate(X_test, Y_test)






