from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import logging



# Function to split time series data
def split_time_series_data(X, y, train_size=0.8,  test_size=0.2):
    assert train_size + test_size == 1.0, "The sum of train, and test sizes must be 1.0"

    n = len(X)
    train_end = int(train_size * n)
    test_end = int((train_size + test_size) * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:], y[train_end:]

    return X_train, y_train, X_test, y_test



def loadData(filepath: str, time_step: int, forecast_horizon: int):
    try:
        #logging.info("Preprocessing SmartCare dataset for lag-llama...")

        # Load CSV file
        df_rr = pd.read_csv(filepath)


        # Convert timestamp to datetime and set as index
        df_rr["timestamp"] = pd.to_datetime(df_rr["timestamp"])
        df_rr.set_index("timestamp", inplace=True)

        # Rename 'datapoint' to 'hr' only if 'datapoint' exists
        if 'datapoint' in df_rr.columns and 'hr' not in df_rr.columns:
            df_rr.rename(columns={"datapoint": "hr"}, inplace=True)

        # Ensure 'hr' is numeric
        df_rr["hr"] = pd.to_numeric(df_rr["hr"], errors='coerce')

        # Drop NaNs and duplicates
        df_rr.dropna(inplace=True)
        df_rr = df_rr[~df_rr.index.duplicated(keep='first')]

        # Scale the data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df_rr)

        # data_scaled= df
        # Prepare the data for LSTM
        def create_dataset(data, time_step=180, forecast_horizon=30):
            X, y = [], []
            for i in range(len(data) - time_step - forecast_horizon + 1):
                # X.append(data.iloc[i:(i + time_step), 0].values)

                X.append(data[i:(i + time_step), 0])
                y.append(data[(i + time_step):(i + time_step + forecast_horizon), 0])
                # y.append(data.iloc[(i + time_step):(i + time_step + forecast_horizon), 0].values)

            return np.array(X), np.array(y)

        X, y = create_dataset(data_scaled, time_step=time_step, forecast_horizon=forecast_horizon)

        # Reshape input to be [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Example usage
        X_train, y_train, X_test, y_test = split_time_series_data(X, y, train_size=0.8, test_size=0.2)

        return X_train, X_test, y_train, y_test, scaler


    except Exception as e:
        logging.error(f"Failed to process file {filepath}: {e}")
        raise