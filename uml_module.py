import os
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def update_forecast_plot(forecast):
    """
    Создание графика прогноза.

    Args:
        forecast (np.ndarray): Прогноз.

    Returns:
        str: Путь к сгенерированному файлу графика.
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(forecast, label='Прогноз')
        plt.title('График прогноза')
        plt.xlabel('Шаги')
        plt.ylabel('Значение')
        plt.legend()
        plt.grid()
        output_path = 'C:\\Users\\User\\Desktop\\МИЭМ\\ВКР\\VKR\\venv\\Scripts\\Static\\forecast_plot.png'
        plt.savefig(output_path)
        plt.close()
        return output_path
    except Exception as e:
        print(f"Ошибка в генерации графика прогноза: {str(e)}")
        return None

def update_uml_diagram():
    """
    Генерация вертикальной UML-диаграммы для отделов компании с использованием PlantUML.

    Returns:
        str: Путь к сгенерированному файлу UML-диаграммы.
    """
    try:
        # Описание UML-диаграммы в формате PlantUML
        uml_code = """
        @startuml
        skinparam monochrome true
        skinparam packageStyle rectangle
        skinparam ranksep 50
        skinparam nodesep 50
        top to bottom direction

        ' Управление (отдельный блок)
        package "Управление" as Management {
          [Генеральный директор] --> [Исполнительное руководство]
        }

        ' Финансово-экономическая группа
        package "Финансово-экономическая группа" as FinanceGroup {
          package "Контрольно-ревизионный отдел" as Audit {
            [Контроль] --> [Ревизия]
          }
          package "Экономический отдел" as Economics {
            [Планирование] --> [Анализ]
          }
          package "Финансовый отдел" as Finance {
            [Бюджетирование] --> [Финансовая отчётность]
          }
          Audit --> Economics : контроль
          Economics --> Finance : экономический анализ
        }

        ' Производственная группа
        package "Производственная группа" as ProductionGroup {
          package "Добыча сырья" as Mining {
            [Добыча] --> [Обработка]
          }
          package "Производство удобрений" as Manufacturing {
            [Производство] --> [Контроль качества]
          }
          Mining --> Manufacturing : поставка сырья
        }

        ' Отдел продаж
        package "Отдел продаж" as Sales {
          package "Управление продажами" as SalesManagement {
            [Продажи] --> [Анализ рынка]
          }
          package "Планово-экономический отдел" as Planning {
            [Планирование продаж] --> [Экономический анализ]
          }
          package "Отдел маркетинга" as Marketing {
            [Маркетинговые исследования] --> [Реклама]
          }
          SalesManagement --> Planning : планирование
          SalesManagement --> Marketing : маркетинговая стратегия
        }

        ' Отдел закупок
        package "Отдел закупок" as Procurement {
          package "Отдел закупок" as ProcurementDept {
            [Закупка сырья] --> [Закупка оборудования]
          }
          package "Отдел логистики" as Logistics {
            [Транспортировка] --> [Складское управление]
          }
          ProcurementDept --> Logistics : логистика закупок
        }

        ' Отдел исследований и разработок
        package "Отдел исследований и разработок" as RND {
          package "Разработка технологий" as TechDev {
            [Исследования] --> [Разработка]
          }
          package "Исследовательский центр" as ResearchCenter {
            [Лаборатории] --> [Тестирование]
          }
          TechDev --> ResearchCenter : тестирование технологий
        }

        ' Взаимосвязи между группами и Управлением
        Management --> FinanceGroup : управляет
        Management --> ProductionGroup : управляет
        Management --> Sales : управляет
        Management --> Procurement : управляет
        Management --> RND : управляет

        ' Упрощённые связи между группами
        ProductionGroup --> Procurement : запросы на сырьё
        ProductionGroup --> Sales : поставка продукции
        Sales --> RND : обратная связь от клиентов
        RND --> ProductionGroup : внедрение технологий
        FinanceGroup --> Sales : финансовая отчётность
        FinanceGroup --> Procurement : бюджет на закупки

        @enduml
        """

        # Сохранение UML-кода в файл
        uml_file_path = 'C:\\Users\\User\\Desktop\\МИЭМ\\ВКР\\VKR\\uml_diagram.puml'
        with open(uml_file_path, 'w', encoding='utf-8') as f:
            f.write(uml_code)

        # Путь к plantuml.jar
        plantuml_jar_path = 'C:\\Users\\User\\Desktop\\МИЭМ\\ВКР\\VKR\\plantuml-1.2025.2.jar'

        # Путь к выходному файлу
        output_path = 'C:\\Users\\User\\Desktop\\МИЭМ\\ВКР\\VKR\\venv\\Scripts\\Static\\uml_diagram.png'

        # Вызов PlantUML для генерации диаграммы
        subprocess.run([
            'java', '-jar', plantuml_jar_path,
            uml_file_path, '-o', os.path.dirname(output_path),
            '-tpng'
        ], check=True)

        # Проверяем, что файл создан
        if not os.path.exists(output_path):
            print("Файл UML-диаграммы не был создан")
            return None

        return output_path
    except Exception as e:
        print(f"Ошибка в генерации UML-диаграммы: {str(e)}")
        return None
