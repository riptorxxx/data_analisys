import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#  _______________ Part 1 ________________

# Save dataset in variable
data = pd.read_csv('airpollution.csv')

print(f"\nПервые 10 строк датасета:\n{data.head(10)}")
print(f"\nИнформация о датасете:")
data.info()
print(f"\nТипы данных в датасете:\n{data.dtypes}")
print(f"\nЧисло строк и столбцов в датасете:\n{data.shape}")
print(f"\nИмена столбцов в датасете:\n{data.columns}")
print(f"\nКол-во пропущенных значений с разбивкой по колонкам:\n{data.isna().sum()}")

#  _______________ Part 1 Answer ________________
# В датасете преобладает тип данных `object` это видно из результата функции `info()`.
# В датасете есть пропущенные значение в столбцах (Country = 427, City = 1)
#   это видно из результата функции `isna().sum()`.


#  _______________ Part 1 Debug ________________

# # Проверка пропущенных значений
# null_check = data.isnull().sum()
# print(null_check)

# # Вывести строки, содержащие пропущенные значения
# print(data[data.isna().any(axis=1)])
#
# # Получить индексы строк с пропущенными значениями
# missing_rows = data[data.isna().any(axis=1)].index
# print(f"\nНомера строк с пропущенными значениями: {missing_rows.tolist()}")
#
# # Для конкретной колонки (например 'Country')
# missing_in_column = data[data['Country'].isna()].index
# print(f"\nНомера строк с пропущенными значениями в колонке 'Country': {missing_in_column.tolist()}")


#  _______________ Part 2 ________________

# Проверка количества дубликатов
duplicates = data.duplicated().sum()
print(f"\nNumber of duplicates: {duplicates}")

# Показать дубликаты
print("\nДубликаты:")
print(data[data.duplicated()])

# Удаление дубликатов
data_no_duplicates = data.drop_duplicates()

# Проверка результата
print(f"\nИсходное количество строк: {len(data)}")
print(f"Количество строк после удаления дубликатов: {len(data_no_duplicates)}")


#  _______________ Visualize ________________

# Выбираем только числовые колонки
numeric_columns = data.select_dtypes(include=['int64'])
print("Found numeric columns:\n", numeric_columns)

# Применяем логарифмирование для нормализации значений
data_log = numeric_columns.apply(lambda x: np.log(1 + x))

# Проверяем результат нормализации:
for column in numeric_columns.columns:
    print(f"\nКолонка: {column}")
    print("Оригинальные значения:", numeric_columns[column].head().values)
    print("После логарифмирования:", data_log[column].head().values)


# Делаем визуализацию по всем AQI в двух видах(boxplot и hstagramm)
for values in data_log:
    # Box plot
    plt.figure(figsize=(10, 5))
    data_log.boxplot(column=values)
    plt.title(f'Box Plot for {values}')
    plt.show()

    # Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(data_log[values], bins=30)
    plt.title(f'Histogram for {values}')
    plt.show()


# Функции для отдельного вызова
# 1. Сравнение всех числовых показателей AQI
def boxplot_all_aqi(data, columns_to_plot):
    plt.figure(figsize=(12, 6))  # размер
    data.boxplot(column=columns_to_plot)
    plt.title('Диаграмма размаха для показателей загрязнения воздуха')
    plt.ylabel('Значения')
    plt.grid(True)
    plt.show()

# 2. Гистограмма распределения AQI Value
def hist_all_aqi(data):
    plt.figure(figsize=(10, 6))
    plt.hist(data['AQI Value'], bins=30)
    plt.title('Распределение AQI Value')
    plt.xlabel('AQI Value')
    plt.ylabel('Частота')
    plt.show()

# 3. Сравнение AQI Value по странам
def boxplot_aqi_in_country(data):
    plt.figure(figsize=(20, 7))
    data.boxplot(column='AQI Value', by='Country')
    plt.xticks(rotation=45)
    plt.title('AQI Value по странам')
    plt.show()

# 4. Scatter plot для корреляции между разными показателями
def scatterplot_between_aqi_params(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['PM2.5 AQI Value'], data['AQI Value'])
    plt.xlabel('PM2.5 AQI Value')
    plt.ylabel('AQI Value')
    plt.title('Корреляция PM2.5 и общего AQI')
    plt.show()

# 5. Средние значения AQI по категориям
def median_aqi_group_by_category(data):
    plt.figure(figsize=(12, 6))
    data.groupby('AQI Category')['AQI Value'].mean().plot(kind='bar')
    plt.title('Средний AQI Value по категориям')
    plt.xticks(rotation=10)
    plt.show()

# boxplot_all_aqi(data, numeric_columns)
# boxplot_aqi_in_country(data)
# hist_all_aqi(data)
# scatterplot_between_aqi_params(data)
# median_aqi_group_by_category(data)



