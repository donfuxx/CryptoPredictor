# https://medium.datadriveninvestor.com/multivariate-time-series-using-rnn-with-keras-7f78f4488679

import inline as inline
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.offsets import DateOffset
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import Huber
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import warnings
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

warnings.filterwarnings("ignore")

tensorflow.keras.backend.clear_session()

# Configuration
CURRENCY = "BTC"
CSV_PATH = f'https://query1.finance.yahoo.com/v7/finance/download/{CURRENCY}-USD?period1=1113417600&period2=7622851200&interval=1d&events=history&includeAdjustedClose=true'
N_INPUT = 7
TRAIN_SPLIT = N_INPUT * 1
N_FEATURES = 6
EPOCHS = 500
PRED_BATCHES = 3
DROPOUT = 0.1
BATCH_SIZE = 32
UNITS = N_INPUT * N_FEATURES * BATCH_SIZE
# UNITS = N_INPUT * 1

# download data
df = pd.read_csv(CSV_PATH, parse_dates=['Date'])
# df = df.drop(columns=['High', 'Low', 'Close', 'Adj Close', 'Volume'])

# Put the month column in the index.
df = df.set_index("Date")

# fill nan values
df = df.fillna(df.mean())

print(df.head())

# # Split data between the training and testing sets.
# train, test = df[:-TRAIN_SPLIT], df[-TRAIN_SPLIT:]
# print(f'\n train: \n {train}')
# print(f'\n test: \n {test}')
#
# # Scale data.
# scaler = MinMaxScaler()
# scaler.fit(df)
# train = scaler.transform(train)
# test = scaler.transform(test)
# print(f'\n train scaled: \n {train}')
# print(f'\n test scaled: \n {test}')

stock_data = df

input_feature = stock_data.iloc[:, [5, 1]].values
input_data = input_feature

# plt.plot(input_feature[:, 0])
# plt.title("Volume of stocks sold")
# plt.xlabel("Time (latest-> oldest)")
# plt.ylabel("Volume of stocks traded")
# plt.show()
#
# plt.plot(input_feature[:, 1], color='blue')
# plt.title("Google Stock Prices")
# plt.xlabel("Time (latest-> oldest)")
# plt.ylabel("Stock Opening Price")
# plt.show()

sc = MinMaxScaler(feature_range=(0, 1))
input_data[:, 0:2] = sc.fit_transform(input_feature[:, :])

lookback = 50

test_size = int(.95 * len(stock_data))
X = []
y = []
for i in range(len(stock_data) - lookback - 1):
    t = []
    for j in range(0, lookback):
        t.append(input_data[[(i + j)], :])
    X.append(t)
    y.append(input_data[i + lookback + 1, 1])

X, y = np.array(X), np.array(y)

y_train = y[:test_size + lookback]
X_train = X[:test_size + lookback]
X = X.reshape(X.shape[0], lookback, 2)
X_train = X_train.reshape(X_train.shape[0], lookback, 2)
print(X.shape)
print(X_train.shape)

X_test = X[test_size + lookback:]
X = X.reshape(X.shape[0], lookback, 2)
X_test = X_test.reshape(X_test.shape[0], lookback, 2)
print(X.shape)
print(X_test.shape)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 2)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train, y_train, epochs=20, batch_size=32)

predicted_value = model.predict(X_test)
print(X_test)
print(len(X_test))
print(predicted_value)
print(len(predicted_value))

plt.figure(figsize=(20, 8))
# plt.plot(input_data[lookback + test_size:test_size + (2 * lookback), 1], color='green')
plt.plot(input_data[(2 * lookback) + test_size + 1:, 1], color='green')
# plt.plot(predicted_value[-lookback:], color='red')
plt.plot(predicted_value, color='red')
plt.legend(['Actual', 'Prediction'], loc='best', fontsize='xx-large')
plt.title("Opening price of stocks sold")
plt.xlabel("Time (latest-> oldest)")
plt.ylabel("Stock Opening Price")
plt.show()

# tensorflow.keras.backend.clear_session()
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(X, y, epochs=20, batch_size=32)
#
# predicted_value = model.predict(X_train[-lookback + test_size: test_size])
#
# # plt.plot(input_data[lookback + test_size:test_size + (2 * lookback), 1], color='green')
# plt.plot(input_data[-lookback + test_size: test_size, 1], color='green')
# # plt.plot(predicted_value[-lookback:], color='red')
# plt.plot(predicted_value, color='red')
# plt.legend(['Actual', 'Prediction'], loc='best', fontsize='xx-large')
# plt.title("Opening price of stocks sold")
# plt.xlabel("Time (latest-> oldest)")
# plt.ylabel("Stock Opening Price")
# plt.show()


# plt.plot(input_data[lookback:test_size + (2 * lookback), 1], color='green')
# plt.plot(predicted_value, color='red')
# plt.legend(['Actual', 'Prediction'], loc='best', fontsize='xx-large')
# plt.title("Opening price of stocks sold")
# plt.xlabel("Time (latest-> oldest)")
# plt.ylabel("Stock Opening Price")
# plt.show()

# def summary(for_model: Model) -> str:
#     summary_data = []
#     for_model.summary(print_fn=lambda x: summary_data.append(x))
#     return '\n'.join(summary_data)
#
#
# def compile_model() -> Model:
#     model = Sequential()
#     model.add(Conv1D(filters=N_INPUT * N_FEATURES, kernel_size=5,
#                      strides=1, padding="causal",
#                      activation="linear",
#                      input_shape=(N_INPUT, N_FEATURES)))
#     # model.add(
#     #     Bidirectional(LSTM(UNITS, activation='linear', input_shape=(N_INPUT, N_FEATURES), return_sequences=True)))
#     # model.add(LSTM(UNITS, activation='linear', input_shape=(N_INPUT, N_FEATURES), return_sequences=True))
#     # model.add(LSTM(UNITS, activation='linear', input_shape=(N_INPUT, N_FEATURES)))
#     # model.add(Dropout(DROPOUT))
#     # model.add(Bidirectional(LSTM(UNITS, activation='linear', return_sequences=True)))
#     # model.add(Dense(UNITS))
#     # model.add(Dropout(DROPOUT))
#     # model.add(Bidirectional(LSTM(UNITS, activation='linear', return_sequences=True)))
#     # model.add(Dropout(DROPOUT))
#     # model.add(Bidirectional(LSTM(UNITS, activation='linear', return_sequences=True)))
#     # model.add(Dropout(DROPOUT))
#     # model.add(Bidirectional(LSTM(UNITS, activation='linear', return_sequences=True)))
#     # model.add(Dropout(DROPOUT))
#     model.add(LSTM(UNITS, activation='linear'))
#     model.add(Dropout(DROPOUT))
#     # model.add(Dense(N_INPUT))
#     # model.add(Dense(UNITS))
#     # model.add(Lambda(lambda x: np.asarray(x).reshape(1024, 1)))
#     model.add(Dense(N_FEATURES, activation='linear'))
#     # model.add(Dense(1, activation='linear'))
#
#     # Build optimizer
#     # model.compile(optimizer='adam', loss='mse')
#     optimizer = SGD(lr=1e-2, momentum=0.9)
#     model.compile(loss=Huber(),
#                   optimizer=optimizer,
#                   metrics=["mae"])
#     return model
#
#
# def create_model_callbacks() -> []:
#     es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=30, verbose=1)
#     rlr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=25, verbose=1)
#     mcp = ModelCheckpoint(filepath='weights.h5', monitor='loss', verbose=1, save_best_only=True,
#                           save_weights_only=True)
#
#     tb = TensorBoard('logs')
#     return [es, rlr, mcp, tb]
#
#
# def plot_learning_rates(lr_model: Model):
#     lr_schedule = LearningRateScheduler(
#         lambda epoch: 1e-6 * 10 ** (epoch / 20))
#     lr_history = lr_model.fit_generator(generator, epochs=100, callbacks=[lr_schedule])
#
#     plt.semilogx(lr_history.history["lr"], lr_history.history["loss"])
#     plt.axis([1e-6, 1e-1, 0, 0.05])
#     plt.show()
#
#
# # Compile and train the model
# model = compile_model()
# generator = TimeseriesGenerator(train, train, length=N_INPUT, batch_size=BATCH_SIZE)
# # plot_learning_rates(model)
# history = model.fit_generator(generator, epochs=EPOCHS, callbacks=create_model_callbacks(), shuffle=False,
#                               use_multiprocessing=True)
#
# # I got the technique below from Caner Dabakoglu here on Medium. In it we are doing a few things:
# #
# #     create an empty list for each of our 12 predictions
# #     create the batch that our model will predict off of
# #     save the prediction to our list
# #     add the prediction to the end of the batch to be used in the next prediction
# pred_list = []
#
#
# def create_validation_batch(n: int):
#     batch = train[-(N_INPUT * n):-(N_INPUT * (n - 1))].reshape((1, N_INPUT, N_FEATURES))
#     for i in range(N_INPUT):
#         pred_list.append(model.predict(batch)[0])
#         batch = np.append(batch[:, 1:, :], [[pred_list[i]]], axis=1)
#
#
# for i in range(PRED_BATCHES):
#     create_validation_batch(PRED_BATCHES - i + 1)
#
# # Now that we have our list of predictions, we need to reverse the scaling we did in the beginning.
# # The code is also creating a dataframe out of the prediction list, which is concatenated with the original dataframe.
# # I did this for plotting. There are many other (better) ways to do this.
# df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
#                           # index=df[-N_INPUT * PRED_BATCHES:].index, columns=['Prediction'])
# index=df[-N_INPUT * PRED_BATCHES:].index,
# columns=['Prediction', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
# df_test = pd.concat([df, df_predict], axis=1)
# df_test = df_test[len(df_test) - N_INPUT * PRED_BATCHES:]
#
# # Plot the predictions
# plt.figure(figsize=(20, 8))
# plt.title(f'{CURRENCY} Price Prediction (loss: {history.history["loss"][-1]}, epochs: {EPOCHS}, range: {N_INPUT})')
# plt.plot(df_test.index, df_test['Open'])
# plt.plot(df_test.index, df_test['Prediction'], color='r')
# plt.figtext(0.7, 0.05, "No financial advice! No warranties! Invest at own risk!", ha="center", fontsize=10,
#             bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
# plt.annotate(summary(model), (0, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top')
# plt.annotate(model.optimizer, (0, 0), (600, -40), xycoords='axes fraction', textcoords='offset points', va='top')
#
# # Predicting Beyond the Dataset
# train = df
# scaler.fit(train)
# train = scaler.transform(train)
# train = train[~pd.isnull(train)]
# train = train.reshape(-1, N_FEATURES)
# generator = TimeseriesGenerator(train, train, length=N_INPUT, batch_size=BATCH_SIZE)
# model.fit_generator(generator, epochs=EPOCHS, callbacks=create_model_callbacks())
#
# pred_list = []
# batch = train[-N_INPUT:].reshape((1, N_INPUT, N_FEATURES))
# for i in range(N_INPUT):
#     pred_list.append(model.predict(batch)[0])
#     batch = np.append(batch[:, 1:, :], [[pred_list[i]]], axis=1)
#
#
# # Create new future dates
# add_dates = [df.index[-1] + DateOffset(days=x) for x in range(0, N_INPUT + 1)]
# future_dates = pd.DataFrame(index=add_dates[1:], columns=df.columns)
#
# # Reverse scale the future prediction
# df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
#                           # index=future_dates[-N_INPUT:].index, columns=['Prediction'])
# index=future_dates[-N_INPUT:].index,
# columns=['Prediction', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
# df_proj = pd.concat([df, df_predict], axis=1)
# df_proj = df_proj[len(df_proj) - N_INPUT * PRED_BATCHES:]
#
# # Plot results
# plt.plot(df_proj.index, df_proj['Prediction'], color='g')
# plt.legend(['Actual', 'Validation', 'Prediction'], loc='best', fontsize='xx-large')
# plt.ylabel('Price')
# plt.xlabel('Date')
#
# plt.savefig(
#     f'plots/{CURRENCY}_price_{pd.to_datetime(df.index[-1]).date()}_{EPOCHS}_{N_INPUT}_{history.history["loss"][-1]}.png')
# plt.show()
