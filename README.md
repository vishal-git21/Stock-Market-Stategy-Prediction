# Analytics-Stock-Market-Prediction

Navigating stock market unpredictability, investment banks offer invaluable expertise in predicting trends for companies seeking to capitalize on market dynamics. This project aims to predict stock prices and provide investment strategies using various machine learning and statistical techniques.

## Table of Contents
1. [Overview](#overview)
2. [Data Description](#data-description)
3. [Dependencies](#dependencies)
4. [Data Preprocessing](#data-preprocessing)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Time Series Analysis](#time-series-analysis)
7. [Modeling](#modeling)
8. [Evaluation](#evaluation)
9. [Prediction and Visualization](#prediction-and-visualization)
10. [Usage](#usage)
11. [Results](#results)
12. [Conclusion](#conclusion)

## Overview
This project involves the prediction of stock prices and the recommendation of investment strategies based on historical stock data. The process includes data preprocessing, exploratory data analysis, time series analysis, modeling using ARIMAX and XGBoost, and visualization of the predicted stock prices and strategies.

## Data Description
The dataset consists of two files:
- `train.csv`: Historical stock data for training the models.
- `test.csv`: Historical stock data for testing the models.

The data includes the following columns:
- `Date`: The date of the stock data.
- `Open`: The opening price of the stock.
- `Close`: The closing price of the stock.
- `Volume`: The trading volume of the stock.
- `Strategy`: The investment strategy ('Buy', 'Sell', 'Hold') for training data.

## Dependencies
The following libraries are required to run the project:
- numpy
- pandas
- matplotlib
- seaborn
- statsmodels
- sklearn
- xgboost

Install the required libraries using pip:
```bash
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn xgboost
``` 

## Data Preprocessing
The data is preprocessed by handling missing values, normalizing features, creating lag features, and calculating rolling statistics.
```bash
import pandas as pd

train_data = pd.read_csv('/kaggle/input/dataset/train.csv')
train_data['Date'] = pd.to_datetime(train_data['Date'])
train_data = train_data.set_index('Date')

``` 
## Exploratory Data Analysis
Exploratory data analysis includes visualizing the distribution of stock prices and volumes, and calculating descriptive statistics.
```bash
import matplotlib.pyplot as plt

plt.hist(train_data['Open'])
plt.title('Distribution of Opening Prices')
plt.xlabel('Open Price')
plt.ylabel('Frequency')
plt.show()

``` 
## Time Series Analysis
Time series analysis involves plotting the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for the stock prices.
```bash
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train_data['Close'], lags=40)
plt.title('Autocorrelation Function')
plt.show()

plot_pacf(train_data['Close'], lags=40)
plt.title('Partial Autocorrelation Function')
plt.show()

``` 
## Modeling
ARIMAX Model
An ARIMAX model is used to predict the closing prices of the stock. Exogenous variables include open price, volume, interaction features, rolling statistics, and lag features.
```bash
from statsmodels.tsa.statespace.sarimax import SARIMAX

exog_vars = ['Open', 'Volume']
arimax_model = SARIMAX(train_data['Close'], order=(0, 1, 1), exog=train_data[exog_vars])
arimax_model_fit = arimax_model.fit()

``` 
XGBoost Classifier
An XGBoost classifier is used to predict the investment strategy. Features include open price, volume, moving average, RSI, VWAP, MACD, Bollinger Bands, and OBV.
```bash
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

features = ['Open', 'Volume', 'Moving_Average', 'RSI', 'VWAP', 'MACD', 'Bollinger_Bands', 'OBV']
X = train_data[features]
y = train_data['Strategy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

``` 
## Evaluation

The Mean Squared Error (MSE) and accuracy of the models are calculated to evaluate their performance.
```bash
mse = ((y_test - y_pred) ** 2).mean()
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test subset: {accuracy}")
print(f"Mean Squared Error: {mse}")

``` 
## Prediction and Visualization
The predicted stock prices and strategies are visualized using matplotlib.
```bash
plt.plot(test_df['Close'], label='Actual Price', color='blue')
plt.plot(test_df['Close'].where(test_df['Strategy'] == 'Buy'), marker='^', color='green', label='Buy Signal', linestyle='')
plt.plot(test_df['Close'].where(test_df['Strategy'] == 'Sell'), marker='v', color='red', label='Sell Signal', linestyle='')
plt.legend()
plt.title('Stock Price and Investment Strategy Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

``` 
## Usage
Run the provided Python scripts to preprocess the data, train the models, make predictions, and visualize the results.
```bash
python preprocess.py
python train.py
python predict.py
python visualize.py

``` 
## Results
The models achieved an accuracy of 85% on the test data. The predicted stock prices and investment strategies are saved in prediction.csv.

## Conclusion
This project demonstrates the application of machine learning and statistical techniques to predict stock prices and recommend investment strategies. The ARIMAX model effectively captures the time series patterns, while the XGBoost classifier provides accurate strategy recommendations.
