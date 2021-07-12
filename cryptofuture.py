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
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History

rcParams.update({'figure.autolayout': True})

warnings.filterwarnings("ignore")

tensorflow.keras.backend.clear_session()

# Configuration
EPOCHS = 1000
DROPOUT = 0.1
BATCH_SIZE = 8
LOOK_BACK = 60
UNITS = LOOK_BACK * 1
VALIDATION_SPLIT = .0
PREDICTION_RANGE = 30
DYNAMIC_RETRAIN = False
USE_SAVED_MODELS = True


def summary(for_model: Model) -> str:
    summary_data = []
    for_model.summary(print_fn=lambda line: summary_data.append(line))
    return '\n'.join(summary_data)


def create_model_callbacks(es_patience: int = 40, lr_patience: int = 30) -> []:
    es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=es_patience, verbose=1)
    rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=lr_patience, verbose=1)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='loss', verbose=1, save_best_only=True,
                          save_weights_only=True)

    tb = TensorBoard('logs')
    return [es, rlr, mcp, tb]


def moving_average(array: [], w: int) -> []:
    return np.concatenate((np.full(w - 1, array[w]), np.convolve(array, np.ones(w), 'valid') / w))


def df_info(name: str, data):
    print(f'\n{name}.shape: {data.shape}')
    print(f'{name}.describe(): {data.describe()}')
    print(f'{name}.head(): {data.head()}')
    print(f'{name}.tail(): {data.tail()}')
    print('\n')


def build_model(n_output: int) -> Model:
    new_model = Sequential()
    new_model.add(Conv1D(filters=LOOK_BACK, kernel_size=5,
                         strides=1, padding="causal",
                         activation="relu",
                         input_shape=(x.shape[1], N_FEATURES)))
    # new_model.add(
    #     Bidirectional(LSTM(units=UNITS, activation='relu', input_shape=(x.shape[1], N_FEATURES), return_sequences=True)))
    # new_model.add(
    #     Bidirectional(LSTM(units=UNITS, activation='relu', input_shape=(x.shape[1], N_FEATURES))))
    # new_model.add(Dropout(DROPOUT))
    # new_model.add(LSTM(units=UNITS, return_sequences=True, input_shape=(x.shape[1], N_FEATURES)))
    new_model.add(Bidirectional(LSTM(units=UNITS, activation='relu', return_sequences=True)))
    # new_model.add(Dropout(DROPOUT))
    new_model.add(Bidirectional(LSTM(units=UNITS, activation='tanh', return_sequences=True)))
    # new_model.add(Dropout(DROPOUT))
    # new_model.add(Bidirectional(LSTM(units=UNITS, activation='relu', return_sequences=True)))
    # new_model.add(Dropout(DROPOUT))
    new_model.add(Bidirectional(LSTM(units=UNITS, activation='linear')))
    new_model.add(Dropout(DROPOUT))
    # new_model.add(LSTM(units=UNITS))
    new_model.add(Dense(units=n_output))
    # new_model.add(Dense(units=N_FEATURES))

    new_model.compile(optimizer='adam', loss='mean_squared_error')
    # optimizer = SGD(lr=1e-1, momentum=0.9)
    # new_model.compile(loss=Huber(),
    #               optimizer=optimizer,
    #               metrics=["mae"])
    return new_model


def fit_model(new_x: [], new_y: [], new_model: Model, epochs: int = EPOCHS, split: float = VALIDATION_SPLIT,
              es_patience: int = 40, lr_patience: int = 30) -> [Model, History]:
    new_model.fit(new_x, new_y, epochs=epochs, batch_size=BATCH_SIZE,
                  callbacks=create_model_callbacks(es_patience, lr_patience),
                  validation_split=split
                  )
    new_model.load_weights(filepath="weights.h5")
    return [new_model, new_model.history]


def get_updated_x(x_last: [], last_prediction: []) -> []:
    # print(f'x_last input values: {x_last[-1]}')

    x_last = np.append(x_last[1:], last_prediction)
    x_last = x_last.reshape(LOOK_BACK, N_FEATURES)
    # print(f'x_last new: {X_last}')
    # print(f'x_last input values new: {x_last[-1]}')

    return np.expand_dims(x_last, axis=0)


def get_stats() -> str:
    return f'loss: {history.history.get("loss")[-1]} \n ' \
           f'loss multi: {history_multi.history.get("loss")[-1]} \n ' \
           f'EPOCHS: {EPOCHS} DYNAMIC_RETRAIN: {DYNAMIC_RETRAIN} \n ' \
           f'UNITS: {UNITS} \n ' \
           f'BATCH_SIZE: {BATCH_SIZE} \n ' \
           f'LOOK_BACK: {LOOK_BACK} \n ' \
           f'VALIDATION_SPLIT: {VALIDATION_SPLIT} \n ' \
           f'DROPOUT: {DROPOUT} \n ' \
           f'N_FEATURES: {N_FEATURES} \n ' \
           f'PREDICTION_RANGE: {PREDICTION_RANGE} '


# Download data
headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"}
df = pd.read_csv("https://www.coingecko.com/price_charts/export/1/usd.csv", parse_dates=['snapped_at'],
                 storage_options=headers)
df.to_csv('data/btc_price.csv')
df = df.fillna(df.mean())

df_info('df', df)

# https://docs.coinmetrics.io/info/metrics
df_coin = pd.read_csv('data/btc_metrics.csv', parse_dates=['date'])
# df_coin = pd.read_csv('https://coinmetrics.io/newdata/btc.csv', parse_dates=['date'],
#                       storage_options=headers)
# df_coin.to_csv('data/btc_metrics.csv')
df_coin = df_coin.drop(columns=['date'])
df_coin = df_coin.fillna(df_coin.mean())
df_coin = df_coin.drop(df_coin.index[:1577])

df_info('df_coin', df_coin)

# Join dataframes
df = pd.concat([df, df_coin], axis=1, join='inner')

# Put the date column in the index.
df = df.set_index("snapped_at")

# add moving averages
open_values = df['price'].to_numpy()
print(f'open_values: {open_values}')

for m in range(10, 210, 10):
    ma = moving_average(open_values, m).tolist()
    print(f'ma_{m}: {ma}')
    df[f'ma_{m}'] = ma

# Fill nan values
df = df.fillna(df.mean())

df_info('df', df)

# input_feature = stock_data
input_feature = df.iloc[:, :].values
N_FEATURES = len(input_feature[0])
# input_feature = stock_data.iloc[:,
#                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]].values
input_data = input_feature

scaler = MinMaxScaler(feature_range=(0, 1))
input_data[:, 0:N_FEATURES] = scaler.fit_transform(input_feature[:, :])

x = []
y = []
y_multi = []
for i in range(len(df) - LOOK_BACK - 1):
    t = []
    for j in range(0, LOOK_BACK):
        t.append(input_data[[(i + j)], :])
    x.append(t)
    y.append(input_data[i + LOOK_BACK, 1])
    y_multi.append(input_data[i + LOOK_BACK, :])

x, y, y_multi = np.array(x), np.array(y), np.array(y_multi)

x_test = x[-2 * LOOK_BACK:]
print(f'x_test: {x_test}')
x = x.reshape(x.shape[0], LOOK_BACK, N_FEATURES)
x_test = x_test.reshape(x_test.shape[0], LOOK_BACK, N_FEATURES)
print(f'x.shape: {x.shape}')
print(f'x_test.shape: {x_test.shape}')

if USE_SAVED_MODELS:
    model = tensorflow.keras.models.load_model("models/model_single")
    model_multi = tensorflow.keras.models.load_model("models/model_multi")
else:
    model = build_model(1)
    model_multi = build_model(N_FEATURES)

model, history = fit_model(x, y, model)
model_multi, history_multi = fit_model(x, y_multi, model_multi)

if VALIDATION_SPLIT == .0:
    model.save('models/model_single')
    model_multi.save('models/model_multi')

y_predict = model.predict(x_test)
y_predict_multi = model_multi.predict(x_test)

for prediction_steps in range(PREDICTION_RANGE):
    x_predict_multi = get_updated_x(x[-1], y_predict_multi[-1])
    y_predict_multi_new = model_multi.predict(x_predict_multi)

    # insert single feature prediction into multi feature prediction
    y_predict_new = model.predict(x_predict_multi)
    y_predict_multi_new[0] = y_predict_new

    # break at extreme values
    if abs(y_predict_multi_new[0, 1]) > 2:
        break

    print(y_predict_multi_new[0, 1])
    # print(f'future_prediction.shape: {y_predict_multi_new.shape}')
    x = np.append(x, x_predict_multi, axis=0)

    y_predict_multi = np.append(y_predict_multi, y_predict_multi_new, axis=0)
    y_predict = np.append(y_predict, [[y_predict_multi_new[0, 1]]], axis=0)
    # print(f'predicted values: {y_predict}')

    # dynamic retrain
    if DYNAMIC_RETRAIN:
        y_multi = np.append(y_multi, y_predict_multi_new, axis=0)
        model_multi, history_multi = fit_model(x, y_multi, model_multi, epochs=5, es_patience=4, lr_patience=3)
        y = np.append(y, y_predict_new[0], axis=0)
        model, history = fit_model(x, y, model, epochs=5, es_patience=4, lr_patience=3)

# Inverse scale value //FIXME inv scaling
# y_predict = scaler.inverse_transform(y_predict)
# y_predict = y_predict[:, 1]
# input_data = scaler.inverse_transform(input_data)
# print(f'y_predict: {y_predict}')
# print(f'input_data: {input_data}')

# y_predict_multi = y_predict_multi[:, 1]
y_predict_multi = y_predict_multi[:, 1]

# Plot graph
plt.figure(figsize=(20, 8))
plt.plot(input_data[-2 * LOOK_BACK:, 1], color='green')
plt.plot(y_predict, color='red')
# plt.plot(y_predict_multi, color='orange')
plt.plot(y_predict_multi[:-PREDICTION_RANGE], color='purple')
plt.axvline(x=len(x_test) - 1, color='blue', label='Prediction split')
plt.axvline(x=len(x_test) - 1 - VALIDATION_SPLIT * len(x), color='blue', label='Validation split')
plt.title(f'BTC Price Prediction (NFA! No Warranties!) - USE_SAVED_MODELS: {USE_SAVED_MODELS}')
plt.legend(['Actual', 'Validation', 'Validation multi', 'Prediction'], loc='best', fontsize='xx-large')
plt.xlabel("Time (latest-> oldest)")
plt.ylabel("Opening Price")
plt.figtext(0.7, 0.05, get_stats(), ha="center", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
plt.annotate(summary(model), (0, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top')
plt.annotate(model.optimizer, (0, 0), (600, -40), xycoords='axes fraction', textcoords='offset points', va='top')

plt.savefig(
    f'plots/BTC_price_{pd.to_datetime(df.index[-1]).date()}_{EPOCHS}_{BATCH_SIZE}_{LOOK_BACK}_{history.history.get("loss")[-1]}.png')
plt.show()
