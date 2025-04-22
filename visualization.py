import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(dates, historical_data, forecasts, column, model_names=None):
    """
    Построение графика с историческими данными и прогнозами.

    Args:
        dates (pd.DatetimeIndex): Даты исторических данных.
        historical_data (np.ndarray): Исторические данные.
        forecasts (list): Список прогнозов от разных моделей.
        column (str): Название столбца (для подписи графика).
        model_names (list): Список названий моделей.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, historical_data, label='Исторические данные', color='blue')

    forecast_dates = pd.date_range(start=dates[-1], periods=len(forecasts[0]) + 1, freq='MS')[1:]
    colors = ['red', 'green', 'purple']
    for i, (forecast, model_name) in enumerate(zip(forecasts, model_names)):
        plt.plot(forecast_dates, forecast, label=f'Прогноз {model_name}', color=colors[i % len(colors)])

    plt.title(f'Прогноз {column}')
    plt.xlabel('Дата')
    plt.ylabel(column)
    plt.legend()
    plt.grid()
    plt.savefig('C:\\Users\\User\\Desktop\\МИЭМ\\ВКР\\VKR\\venv\\Scripts\\Static\\forecast.png')
    plt.close()


def create_metrics_table(dates, data, anomalies, clusters):
    """
    Создание таблицы метрик.

    Args:
        dates (pd.DatetimeIndex): Даты.
        data (np.ndarray): Данные.
        anomalies (np.ndarray): Метки аномалий.
        clusters (np.ndarray): Метки кластеров.

    Returns:
        pd.DataFrame: Таблица метрик.
    """
    df = pd.DataFrame({
        'Дата': dates,
        'Значение': data,
        'Аномалия': anomalies,
        'Кластер': clusters
    })
    return df
