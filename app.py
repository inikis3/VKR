from flask import Flask, render_template, request
import pandas as pd
import os
import logging
from data_processing import load_data, preprocess_data
from ai_module import arima_forecast, prophet_forecast, holt_winters_forecast, detect_anomalies, cluster_data
from uml_module import update_forecast_plot, update_uml_diagram
from visualization import plot_forecast, create_metrics_table

# Настройка логирования
logging.basicConfig(level=logging.INFO, filename='C:\\Users\\User\\Desktop\\МИЭМ\\ВКР\\VKR\\app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Отключаем логи cmdstanpy
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Инициализация Flask с абсолютными путями
app = Flask(__name__,
            template_folder='C:\\Users\\User\\Desktop\\МИЭМ\\ВКР\\VKR\\venv\\Scripts\\Templates',
            static_folder='C:\\Users\\User\\Desktop\\МИЭМ\\ВКР\\VKR\\venv\\Scripts\\Static')

# Логируем путь к папке шаблонов и статических файлов
template_path = app.template_folder
static_path = app.static_folder
logging.info(f"Путь к папке шаблонов: {template_path}")
logging.info(f"Путь к папке статических файлов: {static_path}")


@app.route('/')
def home():
    """
    Главная страница.
    """
    try:
        # Проверяем существование файла index.html
        index_path = os.path.join(template_path, 'index.html')
        if not os.path.exists(index_path):
            logging.error(f"Файл index.html не найден по пути: {index_path}")
            return f"Файл index.html не найден по пути: {index_path}", 500

        logging.info("Попытка рендеринга index.html")
        result = render_template('index.html')
        logging.info("index.html успешно отрендерен")
        return result
    except Exception as e:
        logging.error(f"Ошибка при рендеринге index.html: {str(e)}")
        return f"Ошибка при загрузке главной страницы: {str(e)}", 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Обрабатывает запрос на прогнозирование.
    """
    try:
        logging.info("Начало обработки запроса на прогнозирование")

        # Загрузка данных
        file = request.files['file']
        file_path = 'C:\\Users\\User\\Desktop\\МИЭМ\\ВКР\\VKR\\temp_data.csv'
        file.save(file_path)
        data = load_data(file_path)
        if data is None:
            logging.error("Не удалось загрузить данные из файла: %s", file_path)
            return "Ошибка загрузки данных: проверьте кодировку файла (рекомендуется UTF-8)", 400

        # Получаем выбранный столбец, модель и горизонт прогнозирования из формы
        column = request.form['column']
        model_type = request.form['model']
        steps_ahead = int(request.form['steps_ahead'])
        logging.info(f"Выбран столбец: {column}, модель: {model_type}, горизонт: {steps_ahead}")

        # Проверяем, существует ли столбец в данных
        if column not in data.columns:
            logging.error(f"Столбец {column} отсутствует в данных")
            return f"Столбец {column} отсутствует в данных", 400

        # Проверяем горизонт прогнозирования
        if steps_ahead < 1 or steps_ahead > 120:
            logging.error("Горизонт прогнозирования должен быть от 1 до 120 месяцев")
            return "Горизонт прогнозирования должен быть от 1 до 120 месяцев", 400

        # Предобработка данных
        data, scaler = preprocess_data(data, column)

        # Прогнозы с помощью разных моделей
        forecasts = []
        model_names = []
        if model_type == 'all' or model_type == 'sarima':
            sarima_pred = arima_forecast(data[column], steps_ahead)
            if sarima_pred is not None:
                sarima_pred = sarima_pred.to_numpy()
                sarima_pred = scaler.inverse_transform(sarima_pred.reshape(-1, 1)).flatten()
                forecasts.append(sarima_pred)
                model_names.append('SARIMA')
        if model_type == 'all' or model_type == 'prophet':
            prophet_pred = prophet_forecast(data[column], steps_ahead)
            if prophet_pred is not None:
                prophet_pred = scaler.inverse_transform(prophet_pred.reshape(-1, 1)).flatten()
                forecasts.append(prophet_pred)
                model_names.append('Prophet')
        if model_type == 'all' or model_type == 'holt_winters':
            holt_winters_pred = holt_winters_forecast(data[column], steps_ahead)
            if holt_winters_pred is not None:
                holt_winters_pred = scaler.inverse_transform(holt_winters_pred.reshape(-1, 1)).flatten()
                forecasts.append(holt_winters_pred)
                model_names.append('Holt-Winters')

        if not forecasts:
            logging.error("Не удалось выполнить прогнозирование ни одной моделью")
            return "Ошибка при выполнении прогнозирования", 500

        # Создаем индекс дат для прогноза (на основе первой модели)
        forecast_index = pd.date_range(start=data.index[-1], periods=len(forecasts[0]) + 1, freq='MS')[1:]
        # Преобразуем прогнозы в список словарей для отображения
        forecast_data = []
        for i, forecast in enumerate(forecasts):
            model_forecast = [{'date': date.strftime('%Y-%m-%d'), 'value': value, 'model': model_names[i]}
                              for date, value in zip(forecast_index, forecast)]
            forecast_data.extend(model_forecast)

        # Валидация и кластеризация (на основе исторических данных)
        anomalies = detect_anomalies(data[[column]])
        clusters = cluster_data(data[[column]])

        # Денормализуем исторические данные для графика и метрик
        data_original = scaler.inverse_transform(data[[column]]).flatten()

        # Обновление графика прогноза (используем первый прогноз)
        forecast_plot_path = update_forecast_plot(forecasts[0])
        if forecast_plot_path is None or not os.path.exists(forecast_plot_path):
            logging.error("Не удалось сгенерировать график прогноза или файл не существует")
            forecast_plot_url = None
        else:
            forecast_plot_url = "/forecast_plot.png"

        # Генерация UML-диаграммы
        uml_path = update_uml_diagram()
        if uml_path is None or not os.path.exists(uml_path):
            logging.error("Не удалось сгенерировать UML-диаграмму или файл не существует")
            uml_url = None
        else:
            uml_url = "/uml_diagram.png"

        # Визуализация
        plot_forecast(data.index, data_original, forecasts, column, model_names)
        # Проверяем, существует ли forecast.png
        forecast_path = 'C:\\Users\\User\\Desktop\\МИЭМ\\ВКР\\VKR\\venv\\Scripts\\Static\\forecast.png'
        if not os.path.exists(forecast_path):
            logging.error("Файл forecast.png не существует")
            forecast_url = None
        else:
            forecast_url = "/forecast.png"

        metrics = create_metrics_table(data.index, data_original, anomalies, clusters)

        # Отладка: логируем значения переменных
        for i, (forecast, model_name) in enumerate(zip(forecasts, model_names)):
            logging.info(f"Прогноз {model_name}: {forecast[:5]}...")
        logging.info(f"forecast_url: {forecast_url}")
        logging.info(f"forecast_plot_url: {forecast_plot_url}")
        logging.info(f"uml_url: {uml_url}")
        logging.info(f"metrics: {metrics.to_html()[:100]}...")

        logging.info("Запрос на прогнозирование успешно обработан")

        # Рендеринг шаблона с обработкой ошибок
        try:
            # Проверяем существование файла results.html
            results_path = os.path.join(template_path, 'results.html')
            if not os.path.exists(results_path):
                logging.error(f"Файл results.html не найден по пути: {results_path}")
                return f"Файл results.html не найден по пути: {results_path}", 500

            logging.info("Попытка рендеринга results.html")
            result = render_template('results.html',
                                     forecast_data=forecast_data,
                                     forecast_url=forecast_url,
                                     forecast_plot_url=forecast_plot_url,
                                     uml_url=uml_url,
                                     metrics=metrics.to_html(),
                                     column_name=column,
                                     model_names=model_names)
            logging.info("Шаблон results.html успешно отрендерен")
            return result
        except Exception as e:
            logging.error(f"Ошибка при рендеринге results.html: {str(e)}")
            return f"Ошибка при рендеринге results.html: {str(e)}", 500

    except Exception as e:
        logging.error("Произошла ошибка: %s", str(e))
        return f"Произошла ошибка: {str(e)}", 500


# Тестовый маршрут для проверки шаблона
@app.route('/test-template')
def test_template():
    """
    Тестовый маршрут для проверки рендеринга results.html.
    """
    try:
        # Проверяем существование файла results.html
        results_path = os.path.join(template_path, 'results.html')
        if not os.path.exists(results_path):
            logging.error(f"Файл results.html не найден по пути: {results_path}")
            return f"Файл results.html не найден по пути: {results_path}", 500

        logging.info("Попытка рендеринга results.html в тестовом маршруте")
        result = render_template('results.html',
                                 forecast_data=[{'date': '2025-01-01', 'value': 1000, 'model': 'SARIMA'}],
                                 forecast_url="/forecast.png",
                                 forecast_plot_url="/forecast_plot.png",
                                 uml_url="/uml_diagram.png",
                                 metrics="<p>Тестовая таблица</p>",
                                 column_name="test_column",
                                 model_names=['SARIMA'])
        logging.info("Шаблон results.html успешно отрендерен в тестовом маршруте")
        return result
    except Exception as e:
        logging.error(f"Ошибка при тестовом рендеринге results.html: {str(e)}")
        return f"Ошибка при тестовом рендеринге: {str(e)}", 500


# Маршрут для проверки статических файлов
@app.route('/check-static')
def check_static():
    """
    Проверяет доступность статических файлов.
    """
    try:
        static_files = os.listdir(app.static_folder)
        logging.info(f"Статические файлы в {app.static_folder}: {static_files}")
        return f"Статические файлы в {app.static_folder}: {static_files}"
    except Exception as e:
        logging.error(f"Ошибка при проверке статических файлов: {str(e)}")
        return f"Ошибка при проверке статических файлов: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
