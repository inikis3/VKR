import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

def arima_forecast(data, steps_ahead):
    """
    Прогнозирование временного ряда с помощью SARIMA.

    Args:
        data (pd.Series): Временной ряд для прогнозирования.
        steps_ahead (int): Количество шагов вперёд для прогноза.

    Returns:
        pd.Series: Прогноз на указанное количество шагов.
    """
    try:
        model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=steps_ahead)
        return forecast
    except Exception as e:
        print(f"Ошибка в SARIMA: {str(e)}")
        return None

def prophet_forecast(data, steps_ahead):
    """
    Прогнозирование временного ряда с помощью Prophet.

    Args:
        data (pd.Series): Временной ряд для прогнозирования.
        steps_ahead (int): Количество шагов вперёд для прогноза.

    Returns:
        np.ndarray: Прогноз на указанное количество шагов.
    """
    try:
        df = pd.DataFrame({'ds': data.index, 'y': data.values})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df)
        future = model.make_future_dataframe(periods=steps_ahead, freq='MS')
        forecast = model.predict(future)
        return forecast['yhat'][-steps_ahead:].values
    except Exception as e:
        print(f"Ошибка в Prophet: {str(e)}")
        return None

def holt_winters_forecast(data, steps_ahead):
    """
    Прогнозирование временного ряда с помощью Holt-Winters (Exponential Smoothing).

    Args:
        data (pd.Series): Временной ряд для прогнозирования.
        steps_ahead (int): Количество шагов вперёд для прогноза.

    Returns:
        np.ndarray: Прогноз на указанное количество шагов.
    """
    try:
        model = ExponentialSmoothing(data, seasonal='add', seasonal_periods=12)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps_ahead)
        return forecast.values
    except Exception as e:
        print(f"Ошибка в Holt-Winters: {str(e)}")
        return None

def detect_anomalies(data):
    """
    Обнаружение аномалий с помощью Isolation Forest.

    Args:
        data (pd.DataFrame): Данные для анализа.

    Returns:
        np.ndarray: Метки аномалий (-1 для аномалий, 1 для нормальных точек).
    """
    try:
        model = IsolationForest(contamination=0.1, random_state=42)
        anomalies = model.fit_predict(data)
        return anomalies
    except Exception as e:
        print(f"Ошибка в обнаружении аномалий: {str(e)}")
        return None

def cluster_data(data):
    """
    Кластеризация данных с помощью KMeans.

    Args:
        data (pd.DataFrame): Данные для кластеризации.

    Returns:
        np.ndarray: Метки кластеров.
    """
    try:
        model = KMeans(n_clusters=3, random_state=42)
        clusters = model.fit_predict(data)
        return clusters
    except Exception as e:
        print(f"Ошибка в кластеризации: {str(e)}")
        return None
