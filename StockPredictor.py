import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import os

# Load S&P500 data
sp500 = pd.read_csv("sp500.csv", index_col=0) if os.path.exists("sp500.csv") else yf.Ticker("^GSPC").history(period="max")
sp500.index = pd.to_datetime(sp500.index)
sp500.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()

# Define predictors
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Initialize Random Forest model
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# Train the model
train, test = sp500.iloc[:-100], sp500.iloc[-100:]
model.fit(train[predictors], train["Target"])

# Evaluate precision score on test data
preds = model.predict(test[predictors])
precision = precision_score(test["Target"], preds)
print("Precision Score:", precision)

# Backtest function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

# Enhanced predictors for backtesting
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500["Close"].rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column, trend_column]

sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"], inplace=True)

# Update the model for backtesting
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Updated predict function for backtesting
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = (model.predict_proba(test[predictors])[:, 1] >= 0.6).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Backtest with enhanced predictors
predictions = backtest(sp500, model, new_predictors)

# Evaluate precision score on backtest data
precision_backtest = precision_score(predictions["Target"], predictions["Predictions"])
print("Precision Score (Backtest):", precision_backtest)

# Display target distribution in backtest predictions
print("Target Distribution in Backtest Predictions:")
print(predictions["Target"].value_counts() / predictions.shape[0])

# Display the predictions
print("Predictions:")
print(predictions)
