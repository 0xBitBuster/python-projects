import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

yf.pdr_override()


'''Load Data'''
ticker = 'META'
start_date = dt.datetime(2012, 1, 1)
end_date = dt.datetime(2020, 1, 1)

data = yf.download(ticker, start=start_date, end=end_date)


'''Preprocess Data'''
scaler = MinMaxScaler(feature_range=(0, 1))  # Scale down all values to be between 0 and 1
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])  # Append prices sequence used for predicting the next day value
    y_train.append(scaled_data[x, 0])  # Close price for the next day

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape data to be compatible with model


'''Build Model'''
# Long Short-Term Memory (LTSM) layers are used to recognize patterns over time
# Dropout layers prevent over fitting, they randomly set a fraction of input units to 0 at each update (0.2 = 20%)
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Predict a single continuous value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)


'''Test the model on existing data'''
# Load test data
test_start_date = dt.datetime(2020, 1, 1)
test_end_date = dt.datetime.now()

test_data = yf.download(ticker, start=test_start_date, end=test_end_date)
actual_prices = test_data['Close'].values

# Concatenate training data with the test data
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
# The subset starts from the end of the training data and goes up to the start of the test data
# This is done to ensure that the model inputs do not include any future data points
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make predictions on test data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(scaled_data[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the test predictions
plt.plot(predicted_prices, color='green', label='Predicted')
plt.plot(actual_prices, color='black', label='Actual')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(f'{ticker} Stock Price')
plt.legend()
plt.show()


'''Predict the next day'''
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

print(f'Predicted price for next day: ${prediction[0][0]}')
