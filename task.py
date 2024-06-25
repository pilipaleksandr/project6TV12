import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Price': np.random.normal(0, 1, 100).cumsum() + np.sin(np.linspace(0, 10, 100)) * 10
})
data.set_index('Date', inplace=True)

decomp = seasonal_decompose(data['Price'], model='additive')
trend_component = decomp.trend
seasonal_component = decomp.seasonal
residual_component = decomp.resid

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(data['Price'], color='blue', label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend_component, color='orange', label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal_component, color='green', label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual_component, color='red', label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

arima_model = ARIMA(data['Price'], order=(5, 1, 0))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=6)
print('ARIMA Forecast:', arima_forecast)

sarima_model = SARIMAX(data['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
sarima_results = sarima_model.fit()
data['SARIMA_Forecast'] = sarima_results.predict(start=len(data)-10, end=len(data)+5, dynamic=True)
data[['Price', 'SARIMA_Forecast']].plot(color=['blue', 'purple'])
plt.title('SARIMA Forecast')
plt.show()

adf_result = adfuller(data['Price'])
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])

data['Other_Series'] = np.random.normal(0, 1, 100).cumsum() + np.cos(np.linspace(0, 10, 100)) * 5
var_model = VAR(data[['Price', 'Other_Series']])
var_results = var_model.fit()
print(var_results.summary())

data['trend_index'] = np.arange(len(data))
for i in range(1, 4):
    data[f'lag_{i}'] = data['Price'].shift(i)
data.dropna(inplace=True)

features = data[['trend_index'] + [f'lag_{i}' for i in range(1, 4)]].values
target = data['Price'].values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, shuffle=False)

rf_model = RandomForestRegressor(n_estimators=200)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(data.index[len(data) - len(y_test):], y_test, color='blue', label='Actual')
plt.plot(data.index[len(data) - len(y_test):], rf_predictions, color='red', label='Predicted')
plt.title('Random Forest Predictions')
plt.legend()
plt.show()
