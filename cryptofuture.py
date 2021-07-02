# https://medium.datadriveninvestor.com/multivariate-time-series-using-rnn-with-keras-7f78f4488679

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline
import tensorflow
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

rcParams.update({'figure.autolayout': True})

warnings.filterwarnings("ignore")

tensorflow.keras.backend.clear_session()

# Configuration
CURRENCY = "BTC"
CSV_PATH = f'https://query1.finance.yahoo.com/v7/finance/download/{CURRENCY}-USD?period1=1113417600&period2=7622851200&interval=1d&events=history&includeAdjustedClose=true'
# N_FEATURES = 26
EPOCHS = 1
DROPOUT = 0.1
BATCH_SIZE = 128
LOOK_BACK = 50
UNITS = LOOK_BACK * 3
TEST_SPLIT = .8
PREDICTION_RANGE = LOOK_BACK


def create_model_callbacks() -> []:
    es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=40, verbose=1)
    rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=30, verbose=1)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='loss', verbose=1, save_best_only=True,
                          save_weights_only=True)

    tb = TensorBoard('logs')
    return [es, rlr, mcp, tb]


def moving_average(array: [], w: int) -> []:
    return np.concatenate((np.ones(w - 1), np.convolve(array, np.ones(w), 'valid') / w))


# download data
df = pd.read_csv(CSV_PATH, parse_dates=['Date'])
# df = df.drop(columns=['High', 'Low', 'Close', 'Adj Close', 'Volume'])
# df_coin = pd.read_csv('https://coinmetrics.io/newdata/btc.csv', parse_dates=['date'])


# Put the month column in the index.
df = df.set_index("Date")

# add moving averages
open_values = df['Open'].to_numpy()
print(f'open_values: {open_values}')

for m in range(10, 210, 10):
    ma = moving_average(open_values, m).tolist()
    print(f'ma_{m}: {ma}')
    df[f'ma_{m}'] = ma

# fill nan values
df = df.fillna(df.mean())

print(df.head())
print(df.tail())

stock_data = df

# input_feature = stock_data
input_feature = stock_data.iloc[:, :].values
N_FEATURES = len(input_feature[0])
# input_feature = stock_data.iloc[:,
#                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]].values
input_data = input_feature

scaler = MinMaxScaler(feature_range=(0, 1))
input_data[:, 0:N_FEATURES] = scaler.fit_transform(input_feature[:, :])
# add another model to predict volumes?

test_size = int(TEST_SPLIT * len(stock_data))
x = []
y = []
for i in range(len(stock_data) - LOOK_BACK - 1):
    t = []
    for j in range(0, LOOK_BACK):
        t.append(input_data[[(i + j)], :])
    x.append(t)
    y.append(input_data[i + LOOK_BACK, :])

x, y = np.array(x), np.array(y)

y_train = y[:test_size + LOOK_BACK]
x_train = x[:test_size + LOOK_BACK]
x = x.reshape(x.shape[0], LOOK_BACK, N_FEATURES)
x_train = x_train.reshape(x_train.shape[0], LOOK_BACK, N_FEATURES)
print(f'x.shape: {x.shape}')
print(f'y.shape: {y.shape}')
print(f'x_train.shape: {x_train.shape}')

x_test = x[test_size + LOOK_BACK:]
x = x.reshape(x.shape[0], LOOK_BACK, N_FEATURES)
x_test = x_test.reshape(x_test.shape[0], LOOK_BACK, N_FEATURES)
print(f'x.shape: {x.shape}')
print(f'x_test.shape: {x_test.shape}')

model = Sequential()
model.add(
    Bidirectional(LSTM(units=UNITS, activation='relu', input_shape=(x.shape[1], N_FEATURES), return_sequences=True)))
# model.add(Dropout(DROPOUT))
# model.add(LSTM(units=UNITS, return_sequences=True, input_shape=(x.shape[1], N_FEATURES)))
# model.add(Bidirectional(LSTM(units=UNITS, activation='tanh', return_sequences=True)))
# model.add(Dropout(DROPOUT))
model.add(Bidirectional(LSTM(units=UNITS, activation='linear')))
model.add(Dropout(DROPOUT))
# model.add(LSTM(units=UNITS))
model.add(Dense(units=N_FEATURES))

model.compile(optimizer='adam', loss='mean_squared_error')
# optimizer = SGD(lr=1e-1, momentum=0.9)
# model.compile(loss=Huber(),
#               optimizer=optimizer,
#               metrics=["mae"])
# model.summary()

# history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=create_model_callbacks(),
#                     validation_split=0.1)
history = model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=create_model_callbacks(),
                    # validation_split=0.05
                    )

y_predict = model.predict(x_test)
# print(x_test)
print(len(x_test))
# print(predicted_value)
print(len(y_predict))


def get_updated_x(x_last: [], last_prediction: []) -> []:
    # print(f'x_last input values: {x_last[-1]}')

    x_last = np.append(x_last[1:], last_prediction)
    x_last = x_last.reshape(LOOK_BACK, N_FEATURES)
    # print(f'x_last new: {X_last}')
    # print(f'x_last input values new: {x_last[-1]}')

    return np.expand_dims(x_last, axis=0)


for prediction_steps in range(PREDICTION_RANGE):
    X_predict = get_updated_x(x[-1], y_predict[-1])
    y_predict_new = model.predict(X_predict)
    print(y_predict_new[0, 1])
    # print(f'future_prediction.shape: {y_predict_new.shape}')
    x = np.append(x, X_predict, axis=0)

    y_predict = np.append(y_predict, y_predict_new, axis=0)
    # print(f'predicted values: {y_predict}')

y_predict = y_predict[:, 1]
plt.figure(figsize=(20, 8))
# plt.plot(input_data[lookback + test_size:test_size + (2 * lookback), 1], color='green')
plt.plot(input_data[(2 * LOOK_BACK) + test_size + 1:, 1], color='green')
# plt.plot(predicted_value[-lookback:], color='red')
plt.plot(y_predict, color='red')
plt.legend(['Actual', 'Prediction'], loc='best', fontsize='xx-large')
plt.title("Opening price of stocks sold")
plt.xlabel("Time (latest-> oldest)")
plt.ylabel("Stock Opening Price")

plt.savefig(
    f'plots/{CURRENCY}_price_{pd.to_datetime(df.index[-1]).date()}_{EPOCHS}_{BATCH_SIZE}_{LOOK_BACK}_{history.history["loss"][-1]}.png')
plt.show()
