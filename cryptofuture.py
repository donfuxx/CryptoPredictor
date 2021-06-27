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
N_FEATURES = 2
EPOCHS = 5
DROPOUT = 0.1
BATCH_SIZE = 32
LOOK_BACK = 50
UNITS = LOOK_BACK
TEST_SPLIT = .9


def create_model_callbacks() -> []:
    es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)
    rlr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='loss', verbose=1, save_best_only=True,
                          save_weights_only=True)

    tb = TensorBoard('logs')
    return [es, rlr, mcp, tb]


# download data
df = pd.read_csv(CSV_PATH, parse_dates=['Date'])
# df = df.drop(columns=['High', 'Low', 'Close', 'Adj Close', 'Volume'])

# Put the month column in the index.
df = df.set_index("Date")

# fill nan values
df = df.fillna(df.mean())

print(df.head())

stock_data = df

input_feature = stock_data.iloc[:, [5, 1]].values
input_data = input_feature

scaler = MinMaxScaler(feature_range=(0, 1))
input_data[:, 0:N_FEATURES] = scaler.fit_transform(input_feature[:, :])

test_size = int(TEST_SPLIT * len(stock_data))
X = []
y = []
for i in range(len(stock_data) - LOOK_BACK - 1):
    t = []
    for j in range(0, LOOK_BACK):
        t.append(input_data[[(i + j)], :])
    X.append(t)
    y.append(input_data[i + LOOK_BACK + 1, 1])

X, y = np.array(X), np.array(y)

y_train = y[:test_size + LOOK_BACK]
X_train = X[:test_size + LOOK_BACK]
X = X.reshape(X.shape[0], LOOK_BACK, N_FEATURES)
X_train = X_train.reshape(X_train.shape[0], LOOK_BACK, N_FEATURES)
print(X.shape)
print(X_train.shape)

X_test = X[test_size + LOOK_BACK:]
X = X.reshape(X.shape[0], LOOK_BACK, 2)
X_test = X_test.reshape(X_test.shape[0], LOOK_BACK, N_FEATURES)
print(X.shape)
print(X_test.shape)

model = Sequential()
model.add(LSTM(units=UNITS, return_sequences=True, input_shape=(X.shape[1], N_FEATURES)))
model.add(LSTM(units=UNITS, return_sequences=True))
model.add(LSTM(units=UNITS))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')


history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=create_model_callbacks())

predicted_value = model.predict(X_test)
print(X_test)
print(len(X_test))
print(predicted_value)
print(len(predicted_value))

plt.figure(figsize=(20, 8))
# plt.plot(input_data[lookback + test_size:test_size + (2 * lookback), 1], color='green')
plt.plot(input_data[(2 * LOOK_BACK) + test_size + 1:, 1], color='green')
# plt.plot(predicted_value[-lookback:], color='red')
plt.plot(predicted_value, color='red')
plt.legend(['Actual', 'Prediction'], loc='best', fontsize='xx-large')
plt.title("Opening price of stocks sold")
plt.xlabel("Time (latest-> oldest)")
plt.ylabel("Stock Opening Price")

plt.savefig(
    f'plots/{CURRENCY}_price_{pd.to_datetime(df.index[-1]).date()}_{EPOCHS}_{BATCH_SIZE}_{LOOK_BACK}_{history.history["loss"][-1]}.png')
plt.show()

