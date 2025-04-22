import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path="C:\\Users\\User\\Desktop\\МИЭМ\\ВКР\\VKR\\phosagro_data.csv"):
    """
    Загружает данные из CSV-файла и выполняет базовую очистку.
    """
    try:
        # Попробуем разные кодировки
        encodings = ['utf-8', 'windows-1251', 'latin1']
        for encoding in encodings:
            try:
                data = pd.read_csv(file_path, encoding=encoding)
                print(f"Файл успешно прочитан с кодировкой: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError("Не удалось определить кодировку файла", "", 0, "", "")

        # Удаление пропусков
        data = data.dropna()
        # Удаление дубликатов
        data = data.drop_duplicates()
        # Преобразование столбца date в формат datetime
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        return data
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

def preprocess_data(data, column):
    """
    Нормализует данные для указанного столбца.
    """
    scaler = MinMaxScaler()
    data[[column]] = scaler.fit_transform(data[[column]])
    return data, scaler

    print(f"Индекс после загрузки: {data.index}")
    print(f"Частота индекса: {data.index.freq}")
