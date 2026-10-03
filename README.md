# Optimised Stock Price Prediction

Predicting Apple (AAPL) next day closing price using a hybrid LSTM and SVR model.

## Data preprocessing

Notebook: notebooks/01_preprocessing.ipynb

Dataset: Apple daily stock data from Yahoo Finance, 2015-01-02 to 2026-09-30 (2953 rows). The original file is in data/raw/apple_stock_2015_2026.csv and is not changed.

### What was done

1. Loaded the CSV. The file had two extra header rows (Ticker and Date), so these were skipped and the Date column was set as the index.
2. Checked the data: no missing values, no duplicate dates, dates are in order, and all columns are numeric.
3. Sanity checks: no rows where High is lower than Low, no zero volume, no zero or negative prices.
4. Plotted Close price and Volume to see the trend.
5. Split the data by date, 80% train and 20% test, without shuffling.
   - Train: 2015-01-02 to 2024-05-21 (2362 rows)
   - Test: 2024-05-22 to 2026-09-30 (591 rows)
6. Scaled Open, High, Low, Close and Volume to the range 0 to 1 using MinMaxScaler. The scaler was fitted on the train set only, so some test values go above 1. This is expected because test prices are higher than anything in train.
7. Made sequences using a 60 day window. Each input is the last 60 days of all 5 features and the output is the next day's Close.
8. For the test set, the last 60 days of train were added in front so that all 591 test days get a full window.

### Saved files (data/processed)

- train_scaled.csv and test_scaled.csv: scaled data
- scaler.pkl: the fitted scaler
- sequences.npz: X_train, y_train, X_test, y_test

### Shapes

- X_train: (2302, 60, 5), y_train: (2302,)
- X_test: (591, 60, 5), y_test: (591,)

For SVR the windows can be flattened to 2D, which gives (2302, 300) for train and (591, 300) for test.

### Converting predictions back to dollars

The models predict scaled values (0 to 1), so they need to be converted back to prices using the saved scaler. The scaler was fitted on 5 columns, so the predicted Close is placed in column 3 (Close), the other columns are filled with zeros, and then inverse_transform is applied.

    import numpy as np
    import joblib

    scaler = joblib.load("data/processed/scaler.pkl")

    def inverse_close(scaled_close):
        dummy = np.zeros((len(scaled_close), 5))
        dummy[:, 3] = scaled_close
        return scaler.inverse_transform(dummy)[:, 3]