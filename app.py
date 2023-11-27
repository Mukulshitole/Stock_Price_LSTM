import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
import pandas as pd

# Function to download stock data based on user input
def download_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

# Main Streamlit app
st.title('Stock Data Analysis App')

# Get user input for stock ticker
user_input_ticker = st.text_input('Enter Stock Ticker', 'AAPL')

# Get user input for start and end dates
start_date = st.date_input('Select Start Date', value=pd.to_datetime("2010-01-01"))
end_date = st.date_input('Select End Date', value=pd.to_datetime("2023-01-01"))

# Display date range based on user input
st.subheader(f'Data for {user_input_ticker} from {start_date} to {end_date}')

# Submit button to trigger data download and analysis
if st.button('Submit'):
    # Call the function to download stock data
    df = download_stock_data(user_input_ticker, start_date, end_date)

    # Display the downloaded data
    st.write(f'Data for {user_input_ticker} from {start_date} to {end_date} is:')
    st.write(df.describe())

    # Visualize graph: Closing Price vs Time chart
    st.subheader('Closing Price vs Time chart')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'])
    st.pyplot(fig)

    # Visualize graph: Closing Price vs Time chart with 100MA
    st.subheader('Closing Price vs Time chart with 100MA')
    ma100 = df['Close'].rolling(100).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ma100, label='100MA')
    ax.plot(df['Close'], label='Closing Price')
    ax.legend()
    st.pyplot(fig)

    # Visualize graph: Closing Price vs Time chart with 100MA & 200MA
    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma100 = df['Close'].rolling(100).mean()
    ma200 = df['Close'].rolling(200).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ma100, label='100MA', color='r')
    ax.plot(ma200, label='200MA', color='g')
    ax.plot(df['Close'], label='Closing Price', color='b')
    ax.legend()
    st.pyplot(fig)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

    print(data_training.shape)
    print(data_testing.shape)

    data_training_array = scaler.fit_transform(data_training)

    model = load_model('keras_model.h5')

    past_100_days = data_training.tail(100)

    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i, 0])  # Fix here
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    print(x_test.shape)
    print(y_test.shape)

    # Make predictions
    y_predicted = model.predict(x_test)
    y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1))  # Reshape to 2D

    # Inverse transform test data for plotting
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot results
    st.subheader('Actual vs Predicted Stock Prices')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test, label='Actual Price', color='blue')
    ax.plot(y_predicted, label='Predicted Price', color='red')
    ax.set_title('Actual vs Predicted Stock Prices')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)
