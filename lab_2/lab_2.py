"""
https://github.com/riptorxxx/data_analisys/tree/main
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

data = pd.read_csv('Airpollution.csv')

# 1. Удаяем колонку City
data = data.drop('City', axis=1)
data.info()  # Убеждаемся, что колонка City отсутствует

# 2-3. Выделяем target и features (два отдельных набора данных)
y = data['AQI Value']    # Целевая переменная - индекс качества воздуха
X = data.drop('AQI Value', axis=1)    # Все остальные признаки

# 4. Разделяем на test и train выборки (80% данных идёт в train, 20% в test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Делаем Scaling (все числовые данные приводятся к среднему 0 и стандартному отклонению 1)
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
sc = StandardScaler()
X_train_scaled = pd.DataFrame(sc.fit_transform(X_train[numeric_features]), columns=numeric_features)
X_test_scaled = pd.DataFrame(sc.transform(X_test[numeric_features]), columns=numeric_features)
print(f"\nScaled data sample:\n{X_train_scaled.head()}")

# 6. One-Hot Encoding категориальных данных
# (каждая категориальная переменная превращается в набор бинарных колонок (0 и 1))
categorical_features = X_train.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(X_train[categorical_features])

X_train_encoded = pd.DataFrame(encoder.transform(X_train[categorical_features]),
                               columns=encoder.get_feature_names_out(categorical_features))
X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_features]),
                              columns=encoder.get_feature_names_out(categorical_features))

print(f"\nEncoded data sample:\n{X_train_encoded.head()}")

# 7. Concatenate scaled and encoded data
# получаем единый датафрейм с масштабированными числовыми и закодированными категориальными признаками
X_train_final = pd.concat([X_train_scaled, X_train_encoded], axis=1)
X_test_final = pd.concat([X_test_scaled, X_test_encoded], axis=1)

print(f"\nFinal transformed data:\n{X_train_final.head()}")

# 8. Compare original and transformed data
print(f"\nOriginal data statistics:\n{X_train[numeric_features].describe()}")
print(f"\nScaled data statistics:\n{X_train_scaled.describe()}")

# После проведнных манипуляций над данными датасет стал более удобным для анализа и моделирования.
# Признаки стали сопоставимыми между собой.
