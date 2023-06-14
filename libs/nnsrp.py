import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.family'] = 'Times New Roman'
# plt.style.use('ggplot') # graph style
# plt.rcParams['figure.figsize'] = [12, 9] # graph size

def set_random_seed(seed):
  tf.keras.utils.set_random_seed(seed) 
  # tf.random.set_seed(seed) # This code is no suitable for Mac
  np.random.seed(seed)

def set_hyperparams(params_dict):
  for key, value in params_dict.items():
        globals()[key] = value

def normalization(data):
  scaler = MinMaxScaler()
  data = data.reshape(-1, 1)
  scaler.fit(data)
  data = scaler.transform(data)
  data = data.reshape(-1)
  return data, scaler

def inverse_normalization(data, scaler):
  data = data.reshape(-1, 1)
  data = scaler.inverse_transform(data)
  data = data.reshape(-1)
  return data

def make_data(data):
  # make input data(X) and output data(y)
  X = np.array([])
  y = np.array([])
  for i in range(len(data)-time_lag):
    X = np.append(X, data[i:i+time_lag])
    y = np.append(y, data[i+time_lag])
  X = X.reshape(-1, time_lag, 1)
  y = y.reshape(-1, 1)

  # devide train data from test data
  X_train, X_test = train_test_split(X, test_size=test_size_rate, shuffle=False)
  y_train, y_test = train_test_split(y, test_size=test_size_rate, shuffle=False)

  return X_train, X_test, y_train, y_test

def create_model():
  optimizer = Adam(learning_rate=learning_rate) # optimization method
  model = Sequential()
  model.add(Input(shape=(time_lag, 1)))
  model.add(LSTM(units = hidden_units, 
                activation = activation_func,
                kernel_initializer = RandomNormal(),
                recurrent_initializer = "orthogonal"))
  model.add(Dense(1, activation='linear', kernel_initializer=RandomNormal()))
  model.compile(optimizer=optimizer, loss='mean_squared_error')

  return model

def train_model(model, X_train, y_train, verbose=0):
  callbacks = []
  # early stopping
  if validation and earlystopping:
    es = EarlyStopping(monitor="val_loss", patience=30, verbose=verbose)   # stop learning if model does not improve "patience" time in a row
    callbacks.append(es)

  # save best model
  if validation and best_model:
    modelcheckpoint = ModelCheckpoint(filepath = "best_model.h5",
                                    monitor = "val_loss", 
                                    verbose = verbose, 
                                    save_best_only = True, 
                                    mode = "min")
    callbacks.append(modelcheckpoint)
                                  
  # model learning
  if validation:
    history = model.fit(X_train, y_train, 
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=val_size_rate,
                    verbose=verbose,
                    callbacks=callbacks,
                    shuffle=False)
  else:
    history = model.fit(X_train, y_train, 
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose,
                    callbacks=callbacks,
                    shuffle=False)
  
  return history

def plot_learning_history(history):
  plt.plot(np.array(history.epoch)+1, history.history["loss"], label="Train Loss")
  if validation:
    plt.plot(np.array(history.epoch)+1, history.history['val_loss'], label="Valid Loss")
  plt.title("Model Loss")
  plt.ylabel("Loss")
  plt.xlabel("Epochs")
  plt.ylim([0,0.1])
  plt.legend()
  plt.show()

def print_best_epoch(history):
  min_loss = min(history.history["val_loss"])
  best_epoch = history.history["val_loss"].index(min(history.history["val_loss"]))+1
  print("best epoch:", best_epoch)
  print("minimum validation loss:", min_loss)

def predict_train_data(model, X_train, verbose=0):
  y_train_pred = model.predict(X_train, verbose=verbose)
  return y_train_pred

def predict_test_data(model, X_test, verbose=0):
  y_test_pred = np.array([])
  input = np.array([X_test[0]])
  for i in range(len(X_test)):
    pred = model.predict(input, verbose=verbose)
    y_test_pred = np.append(y_test_pred, pred)
    input = np.append(input, pred)[1:]
    input = input.reshape(1, -1, 1)
  return y_test_pred

def plot_result(y_train, y_test, y_train_pred, y_test_pred):

  val_size = int(len(y_train) * val_size_rate) if validation else 0
  # plot train data prediction
  plt.plot(np.arange(time_lag+1, time_lag+1+len(y_train_pred)-val_size), y_train_pred[:len(y_train_pred)-val_size], color="steelblue", marker='o', label="Predicted-train")
  plt.plot(np.arange(time_lag+1, time_lag+1+len(y_train_pred)-val_size), y_train[:len(y_train_pred)-val_size], color="firebrick", marker='o', label="Actual-train")

  # plot validation data prediction
  plt.plot(np.arange(time_lag+1+len(y_train_pred)-val_size, time_lag+1+len(y_train_pred)), y_train_pred[-val_size:], color="orangered", marker='o', label="Predicted-val")
  plt.plot(np.arange(time_lag+1+len(y_train_pred)-val_size, time_lag+1+len(y_train_pred)), y_train[-val_size:], color="mediumpurple", marker='o', label="Actual-val") 

  # plot test data prediction
  plt.plot(np.arange(time_lag+1+len(y_train_pred), time_lag+1+len(y_train_pred)+len(y_test_pred)), y_test_pred, color="limegreen", marker='o', label="Predicted-test")
  plt.plot(np.arange(time_lag+1+len(y_train_pred), time_lag+1+len(y_train_pred)+len(y_test_pred)), y_test, color="sienna", marker='o', label="Actual-test")  
  plt.xlabel("Test Date")
  plt.ylabel("Fault")
  plt.grid()
  plt.legend()
  plt.show()

def plot_only_testdata(y_test, y_test_pred):
  plt.plot(y_test_pred, color="limegreen", marker='o', label="Predicted")
  plt.plot(y_test, color="sienna", marker='o', label="Actual")
  plt.xlabel("Test Date")
  plt.ylabel("Fault")
  plt.grid()
  plt.legend()
  plt.show()

def print_mse(y_test, y_test_pred):
  mse = mean_squared_error(y_test, y_test_pred)
  print("MSE", mse)
  print("RMSE", math.sqrt(mse))

def get_best_params_with_gridsearch(X_train, y_train, param_grid, create_model, n_splits=3, verbose=1):
  # time series cross-validation
  tscv = TimeSeriesSplit(n_splits).split(X_train)

  # processing for grid search
  model = KerasRegressor(build_fn=create_model, verbose=0)
  grid = GridSearchCV(estimator=model, 
                      param_grid=param_grid, 
                      cv=tscv,
                      refit=True,
                      scoring="neg_root_mean_squared_error",
                      verbose=verbose)

  # training the model
  grid_result = grid.fit(X_train, y_train)

  result = pd.DataFrame(grid_result.cv_results_)

  # display and return the result
  display(result)
  return result

def predictor(data, params_dict, verbose=0, graph_plot=False):
  # set hyperparams as global variable
  set_hyperparams(params_dict)

  # normalize data
  data, scaler = normalization(data)

  # devide train data and test data
  X_train, X_test, y_train, y_test = make_data(data)

  # create NN model
  model = create_model() 
  
  # train NN model
  history = train_model(model, X_train, y_train, verbose=verbose)

  # plot history graph
  if graph_plot:
    plot_learning_history(history)

  # predict
  y_train_pred = predict_train_data(model, X_train, verbose=verbose)
  y_test_pred = predict_test_data(model, X_test, verbose=verbose)

  # inverse normalization
  y_train = inverse_normalization(y_train, scaler)
  y_test = inverse_normalization(y_test, scaler)
  y_train_pred = inverse_normalization(y_train_pred, scaler)
  y_test_pred = inverse_normalization(y_test_pred, scaler)

  if graph_plot:
    # plot predicted value
    plot_result(y_train, y_test, y_train_pred, y_test_pred)
    plot_only_testdata(y_test, y_test_pred)

    print_mse(y_test, y_test_pred)
  return y_test_pred