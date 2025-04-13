import os.path
from typing import Tuple, List

import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO

from config import DATA_FILE, MODEL_FILE
from secure import SecureCSVValidator


# Запуск через команду в консоли `uvicorn lab_3:app --reload`
app = FastAPI()


class LaptopPricePredictor:
    """Класс для обработки данных и предсказания цен на ноутбуки"""

    def __init__(self):
        self.validator = SecureCSVValidator()
        self.model = None
        self.numeric_features = None
        self.categorical_features = None

    def load_data(self, data_file_path: str) -> pd.DataFrame:
        """Загрузка данных из CSV файла"""
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Файл {data_file_path} не найден!")
        return pd.read_csv(data_file_path)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Подготовка данных: разделение на признаки и целевую переменную"""
        x = df.drop(columns=['Price'])
        y = df['Price']
        return x, y

    def get_feature_types(self, x: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Определение типов признаков"""
        numeric_features = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = x.select_dtypes(include=['object']).columns.tolist()
        return numeric_features, categorical_features

    @staticmethod
    def create_numeric_transformer() -> Pipeline:
        """Создание пайплайна для числовых признаков"""
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])

    @staticmethod
    def create_categorical_transformer() -> Pipeline:
        """Создание пайплайна для категориальных признаков"""
        return Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

    def create_preprocessor(self) -> ColumnTransformer:
        """Создание препроцессора"""
        return ColumnTransformer([
            ('num', self.create_numeric_transformer(), self.numeric_features),
            ('cat', self.create_categorical_transformer(), self.categorical_features)
        ])

    @staticmethod
    def create_model() -> XGBRegressor:
        """Создание модели"""
        return XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

    def evaluate_model(self, x_test: pd.DataFrame, y_test: pd.Series) -> float:
        """Оценка качества модели"""
        y_pred = self.model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE: {mse}")
        return mse

    def save_model(self, model_path: str):
        """Сохранение модели"""
        joblib.dump(self.model, model_path)

    def load_data_and_train_model(self):
        """Основная функция: Загрузка данных и обучение модели"""

        # Загрузка и подготовка данных
        df = self.load_data(DATA_FILE)
        x, y = self.prepare_data(df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Определение типов признаков
        self.numeric_features, self.categorical_features = self.get_feature_types(x)

        # Создание и обучение пайплайна
        self.model = Pipeline([
            ('preprocessor', self.create_preprocessor()),
            ('model', self.create_model())
        ])

        self.model.fit(x_train, y_train)
        self.evaluate_model(x_test, y_test)
        self.save_model(MODEL_FILE)

    def predict(self, input_data: pd.DataFrame) -> pd.Series:
        """Предсказание цен на новых данных"""
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите load_data_and_train_model()")
        return self.model.predict(input_data)


# Инициализация модели при старте приложения
predictor = LaptopPricePredictor()
predictor.load_data_and_train_model()


@app.post("/predict")
async def predict_price(file: UploadFile = File(...)):
    """
    Эндпоинт для предсказания цен на ноутбуки

    Принимает CSV файл с характеристиками ноутбуков,
    возвращает предсказанные цены
    """
    try:
        # Валидация файла
        validator = SecureCSVValidator()
        validated_data = await validator.validate_upload(file)

        # Шифрование данных
        encrypted_data = validator.security_processor.encrypt_column(validated_data)

        # Предсказание
        predictions = predictor.predict(encrypted_data)

        # Демонстрация расшифровки
        decrypted_samples = validator.security_processor.decrypt_sample(encrypted_data)
        print("Decrypted samples:", decrypted_samples)

        return {"predictions": predictions.tolist()}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


# # Загрузка данных и обучение модели
# def load_data_and_train_model():
#     file_path = './laptop_price.csv'
#     model_path = './laptop_price_model.pkl'
#     df = pd.read_csv(file_path)
#     a = df.drop(columns=['Price'])
#     b = df['Price']
#
#     a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)
#
#     numeric_features = a.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     categoric_features = a.select_dtypes(include=['object']).columns.tolist()
#
#     numeric_transformer = Pipeline([
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler', StandardScaler())
#     ])
#
#     categoric_transformer = Pipeline([
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('encoder', OneHotEncoder(handle_unknown='ignore'))
#     ])
#
#     preprocessor = ColumnTransformer([
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categoric_transformer, categoric_features)
#     ])
#
#     # Создание и обучение пайплайна
#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5))
#     ])
#     pipeline.fit(a_train, b_train)
#
#     # Оценка модели
#     b_pred = pipeline.predict(a_test)
#     print(f"Test MSE: {mean_squared_error(b_test, b_pred)}")
#
#     joblib.dump(pipeline, model_path)
#
#     return pipeline
#

