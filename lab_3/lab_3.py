import os.path
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


app = FastAPI()


def load_data(data_file_path: str) -> pd.DataFrame:
    """Загрузка данных из CSV файла"""
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Файл {data_file_path} не найден!")
    return pd.read_csv(data_file_path)


def prepare_data(df: pd.DataFrame) -> tuple:
    """Подготовка данных: разделение на признаки и целевую переменную"""
    x = df.drop(columns=['Price'])
    y = df['Price']
    return x, y


def get_feature_types(x: pd.DataFrame) -> tuple:
    """Определение типов признаков"""
    numeric_features = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = x.select_dtypes(include=['object']).columns.tolist()
    return numeric_features, categorical_features


def create_numeric_transformer() -> Pipeline:
    """Создание пайплайна для числовых признаков"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])


def create_categorical_transformer() -> Pipeline:
    """Создание пайплайна для категориальных чисел"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])


def create_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Создание препроцессора"""
    return ColumnTransformer([
        ('num', create_numeric_transformer(), numeric_features),
        ('cat', create_categorical_transformer(), categorical_features)
    ])


def create_model() -> XGBRegressor:
    """Создание модели"""
    return XGBRegressor(n_estimator=100, learning_rate=0.1, max_depth=5)


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse}")
    return mse


def save_model(model: Pipeline, model_path: str):
    """Сохранение модели"""
    joblib.dump(model, model_path)


def load_data_and_train_model():
    """Основная функция: Загрузка данных и обучение модели"""

    # Конфиг
    data_file = './laptop_price.csv'
    model_path = './laptop_price_model.pkl'

    # Загрузка и подготовка данных
    df = load_data(data_file)
    x, y = prepare_data(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Создание пайплайна
    numeric_features, categorical_features = get_feature_types(x)
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    model = create_model()

    pipline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Обучение и оценка
    pipline.fit(x_train, y_train)
    evaluate_model(pipline, x_test, y_test)
    save_model(pipline, model_path)

    return pipline


ml_model = load_data_and_train_model()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        uploaded_data = pd.read_csv(BytesIO(await file.read()))
        predictions = ml_model.predict(uploaded_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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

