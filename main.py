import pandas as pd
import numpy as np
from data_processing import load_data, preprocess_data, prepare_time_series
from ai_module import train_lstm_model, lstm_forecast, arima_forecast, detect_anomalies, cluster_data
from uml_module import update_uml_diagram
from visualization import plot_forecast, create_metrics_table

def main():
    # Загрузка данных
    data = load_data('C:\\Users\\User\\Desktop\\МИЭМ\\ВКР\\VKR\\phosagro_data.csv')
    if data is None:
        return

    column = 'emissions'
    timesteps = 12
    steps_ahead = 48

    # Предобработка
    data, scaler = preprocess_data(data, column)

    # Прогнозирование
    X, y = prepare_time_series(data, column, timesteps)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = train_lstm_model(X, y, timesteps, 1, epochs=10)
    forecast = lstm_forecast(model, data[column].values, timesteps, steps_ahead)
    forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()

    arima_pred = arima_forecast(data[column], steps_ahead)
    arima_pred = scaler.inverse_transform(arima_pred.reshape(-1, 1)).flatten()

    # Валидация и кластеризация
    anomalies = detect_anomalies(data[[column]])
    clusters = cluster_data(data[[column]])

    # Обновление UML
    uml_url = update_uml_diagram(forecast)
    print(f"UML-диаграмма: {uml_url}")

    # Визуализация
    plot_forecast(data, forecast, column)
    metrics = create_metrics_table(data[column], anomalies, clusters)
    print(metrics)

if __name__ == "__main__":
    main()
