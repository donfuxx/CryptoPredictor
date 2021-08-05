# https://medium.datadriveninvestor.com/multivariate-time-series-using-rnn-with-keras-7f78f4488679

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline
import tensorflow
from matplotlib import rcParams
from pandas.tseries.offsets import DateOffset
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
BATCH_SIZE = 128
LOOK_BACK = 100
UNITS = LOOK_BACK * 1
VALIDATION_SPLIT = .01
PREDICTION_RANGE = LOOK_BACK
DYNAMIC_RETRAIN = False
USE_SAVED_MODELS = False
SAVE_MODELS = False


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
    # new_model.add(Dropout(DROPOUT))
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
           f'val_loss: {history_val.history.get("val_loss")[-1]} \n ' \
           f'EPOCHS: {EPOCHS} DYNAMIC_RETRAIN: {DYNAMIC_RETRAIN} \n ' \
           f'UNITS: {UNITS} \n ' \
           f'BATCH_SIZE: {BATCH_SIZE} \n ' \
           f'LOOK_BACK: {LOOK_BACK} \n ' \
           f'VALIDATION_SPLIT: {VALIDATION_SPLIT} \n ' \
           f'DROPOUT: {DROPOUT} \n ' \
           f'N_FEATURES: {N_FEATURES} \n ' \
           f'PREDICTION_RANGE: {PREDICTION_RANGE} '


def predict(model: Model, x: [], x_test: [], y:[], prediction_range: int = PREDICTION_RANGE) -> []:
    y_predict = model.predict(x_test)
    for prediction_steps in range(prediction_range):
        x_predict = get_updated_x(x[-1], y_predict[-1])
        y_predict_new = model.predict(x_predict)

        # break at extreme values
        if abs(y_predict_new[0, 1]) > 2:
            break

        print(y_predict_new[0, 1])
        x = np.append(x, x_predict, axis=0)

        y_predict = np.append(y_predict, y_predict_new, axis=0)
        # y_predict = np.append(y_predict, [ct}')

        # dynamic retrain
        if DYNAMIC_RETRAIN:
            y = np.append(y, y_predict_new, axis=0)
            model, history = fit_model(x, y, model, epochs=5, es_patience=4, lr_patience=3)
            # y = np.append(y, y_predict_new[, model, epochs=5, es_patience=4, lr_patience=3)

    return y_predict


# Download data
headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"}
df = pd.read_csv("https://www.coingecko.com/price_charts/export/1/usd.csv", parse_dates=['snapped_at'],
                 storage_options=headers)
df.to_csv('data/btc_price.csv')
df = df.fillna(df.mean())
dates = df.iloc[:, [0]].values

df_info('df', df)

# https://docs.coinmetrics.io/info/metrics
# df_coin = pd.read_csv('https://coinmetrics.io/newdata/btc.csv', parse_dates=['date'],
#                       storage_options=headers)
# df_coin.to_csv('data/btc_metrics.csv')
df_coin = pd.read_csv('data/btc_metrics.csv', parse_dates=['date'])
df_coin = df_coin.drop(columns=['date'])
df_coin = df_coin.fillna(df_coin.mean())
df_coin = df_coin.drop(df_coin.index[:1577])
df_coin.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
df_coin['index'] = df_coin['index'] - 1577
df_coin = df_coin.set_index('index')

df_info('df_coin', df_coin)

# Join dataframes
df = df.join(df_coin)

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

input_feature = df.iloc[:, :].values
N_FEATURES = len(input_feature[0])
input_data = input_feature.copy()

scaler = MinMaxScaler(feature_range=(0, 1))
input_data[:, 0:N_FEATURES] = scaler.fit_transform(input_feature[:, :])

x = []
y = []
for i in range(len(df) - LOOK_BACK - 1):
    t = []
    for j in range(0, LOOK_BACK):
        t.append(input_data[[(i + j)], :])
    x.append(t)
    y.append(input_data[i + LOOK_BACK, :])

x, y = np.array(x), np.array(y)

x_test = x[-2 * LOOK_BACK:]
x = x.reshape(x.shape[0], LOOK_BACK, N_FEATURES)
x_test = x_test.reshape(x_test.shape[0], LOOK_BACK, N_FEATURES)
print(f'x.shape: {x.shape}')
print(f'x_test.shape: {x_test.shape}')

if USE_SAVED_MODELS:
    model_val = tensorflow.keras.models.load_model("models/model_val")
    model = tensorflow.keras.models.load_model("models/model")
else:
    model_val = build_model(N_FEATURES)
    model = build_model(N_FEATURES)

model_val, history_val = fit_model(x, y, model_val)
tensorflow.keras.backend.clear_session()
model, history = fit_model(x, y, model, split=0)

if SAVE_MODELS:
    model.save('models/model')

y_predict_val = predict(model_val, x, x_test, y, prediction_range=0)
y_predict = predict(model, x, x_test, y)

# Inverse scale value
y_predict = scaler.inverse_transform(y_predict)
y_predict = y_predict[:, 0]

y_predict_val = scaler.inverse_transform(y_predict_val)
y_predict_val = y_predict_val[:, 0]

plot_dates = dates[-2 * LOOK_BACK:]

add_dates = [dates[-1] + DateOffset(days=x) for x in range(0, PREDICTION_RANGE + 1)]

predict_dates = np.concatenate([plot_dates[:-1], add_dates])

# Plot graph
plt.figure(figsize=(20, 8))
plt.plot(dates[-2 * LOOK_BACK:, 0], input_feature[-2 * LOOK_BACK:, 0], color='green', label='Actual')
plt.plot(predict_dates[:-PREDICTION_RANGE], y_predict_val, color='orange', label='Validation')
plt.plot(predict_dates, y_predict, color='red', label='Prediction')
plt.axvline(dates[-1, 0], color='blue', label='Prediction split')
if VALIDATION_SPLIT > 0:
    plt.axvline(dates[int(-1 - VALIDATION_SPLIT * len(x)), 0], color='purple', label='Validation split')
plt.title(f'BTC Price Prediction (NFA! No Warranties!) - USE_SAVED_MODELS: {USE_SAVED_MODELS}')
plt.legend(loc='best', fontsize='xx-large')
plt.xlabel("Time (latest-> oldest)")
plt.ylabel("Opening Price")
plt.figtext(0.7, 0.05, get_stats(), ha="center", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
plt.annotate(summary(model), (0, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top')
plt.annotate(model.optimizer, (0, 0), (600, -40), xycoords='axes fraction', textcoords='offset points', va='top')

plt.savefig(
    f'plots/BTC_price_{pd.to_datetime(df.index[-1]).date()}_{EPOCHS}_{BATCH_SIZE}_{LOOK_BACK}_{history.history.get("loss")[-1]}.png')
plt.show()
