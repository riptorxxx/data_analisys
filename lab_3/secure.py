import pandas as pd
import io
import re
import base64

from abc import ABC, abstractmethod
from typing import Dict, Any
from fastapi import UploadFile, HTTPException
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from lab_3.config import MAX_FILE_SIZE, ALLOWED_COLUMNS
from logger import logger


class DataValidator(ABC):
    """Абстрактный базовый класс для валидации данных"""

    @abstractmethod
    def validate(self, data: Any) -> None:
        """Основной метод валидации"""
        pass


class SecureCSVValidator(DataValidator):
    """Конкретная реализация валидации CSV файлов
        - _validate_file_size: проверяем размер файла. Защита от DDoS.
        - _read_file_contents: защищает от чтения битых файлов.
        - _check_binary_data: защита от исполняемого бинарного кода
        - _validate_structure: проверяем структуру CSV файла.
        - _validate_content: проверяем типы данных и ищем подозрительные вхождения. Защита от XSS.


    """

    def __init__(self):
        self.max_file_size = MAX_FILE_SIZE                            # 5MB
        self.allowed_mime_types = ["text/csv", "application/csv"]
        self.allowed_columns = ALLOWED_COLUMNS              # Допустимые колонки
        self.security_processor = DataSecurityProcessor()

    async def validate_upload(self, file: UploadFile) -> pd.DataFrame:
        """Полный цикл валидации загружаемого файла"""
        await self._validate_file_size(file)
        contents = await self._read_file_contents(file)
        self._check_binary_data(contents)
        await self._validate_mime_type(contents, file.filename)
        df = self._parse_csv(contents)

        if df.empty:
            raise HTTPException(400, "No valid data found in CSV file")

        self._check_csv_injection(df)
        self._validate_structure(df)
        self._validate_content(df)

        df = self.security_processor.encrypt_column(df)
        df = self.security_processor.hash_price(df)

        # Проверка шифрования
        encrypted_samples = df[['RAM_Size', 'RAM_encrypted']].head().to_dict()
        logger.info("Encrypted samples: %s", encrypted_samples)

        # Проверка дешифровки
        decrypted_samples = self.security_processor.decrypt_sample(df)
        logger.info("Decrypted samples: %s", decrypted_samples)

        # Проверка хеширования
        hash_samples = df[['Price', 'Price_hash']].head().to_dict()
        logger.info("Hash samples: %s", hash_samples)

        return df

    async def _validate_file_size(self, file: UploadFile):
        """Проверка размера файла"""
        file.file.seek(0, 2)  # указатель в конец файла (whence: 0 - началао, 1 - текущая поз. 2 - конец)
        file_size = file.file.tell()  # вернёт размер файла относительно указателя
        file.file.seek(0)  # указатель в начало файла
        if file_size > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {self.max_file_size} bytes"
            )

    async def _read_file_contents(self, file: UploadFile) -> bytes:
        """Безопасное чтение содержимого файла"""
        try:
            return await file.read()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading file: {str(e)}"
            )

    async def _validate_mime_type(self, contents: bytes, filename: str):
        """Общая проверка CSV-формата"""
        # 1. Проверка расширения
        if not filename.lower().endswith('.csv'):
            raise HTTPException(400, "File must have .csv extension")

        # 2. Проверка текстового содержания и структуры
        try:
            text = contents[:2048].decode('utf-8')  # Чуть больше данных для анализа
            if not self._looks_like_csv(text):  # Новая улучшенная проверка
                raise ValueError
        except (UnicodeDecodeError, ValueError):
            raise HTTPException(400, "Invalid CSV format")

    def _looks_like_csv(self, text: str) -> bool:
        """Проверяет, что текст имеет структуру CSV"""
        patterns = [
            # 1. Проверка чередования разделителей и данных
            r'([^,\n\r"]+,){2,}[^,\n\r"]+(\r?\n|$)',            # Для запятых
            r'([^;\n\r"]+;){2,}[^;\n\r"]+(\r?\n|$)',            # Для точек с запятой
            r'([^\t\n\r"]+\t){2,}[^\t\n\r"]+(\r?\n|$)'          # Для табуляции
        ]

        # 2. Проверка хотя бы 2 строк с разделителями
        return any(
            re.search(pattern, text, re.MULTILINE)
            for pattern in patterns
        ) and text.count('\n') >= 1                             # Минимум 2 строки (заголовок + данные)

    def _check_binary_data(self, contents: bytes):
        """Поиск бинарных данных в файле"""
        if b'\x00' in contents[:1024]:  # Null-байты
            raise HTTPException(400, "Binary data detected")

        # Проверка на base64-encoded данные
        if re.search(rb'[A-Za-z0-9+/=]{50,}', contents):
            raise HTTPException(400, "Suspicious base64 data")

    def _parse_csv(self, contents: bytes) -> pd.DataFrame:
        """Парсинг CSV с обработкой ошибок"""
        try:
            return pd.read_csv(
                io.BytesIO(contents),
                engine='python',  # Более безопасный парсер
                dtype='object'  # Чтение всех данных как строк изначально
            )
        except pd.errors.EmptyDataError:
            raise HTTPException(400, "The file is empty")
        except pd.errors.ParserError:
            raise HTTPException(400, "Invalid CSV structure")
        except Exception as e:
            raise HTTPException(400, f"Failed to parse CSV: {str(e)}")

    def _check_csv_injection(self, df: pd.DataFrame):
        """Расширенная проверка CSV-инъекций без групп захвата"""
        dangerous_patterns = [
            r'^\s*[=+@-]',                                  # Стартовые символы с пробелами
            r'^\s*\"\s*[=+@-]',                             # Символы после кавычек
            r'\b(?:IMPORT|EXPORT|HYPERLINK)\b',             # (?:...) - non-capturing group for pandas (err)
            r'=\d{10,}',                                    # Длинные числовые выражения
            r'@\w+\.\w+',                                   # Электронные адреса в начале
            r'!\w+',                                        # Ссылки на ячейки в Excel
            r'^\s*\t\s*[=+@-]'                              # Символы после табуляции
        ]

        for col in df.select_dtypes(include=['object']).columns:
            col_data = df[col].astype(str).str.strip()  # Удаление пробелов
            for pattern in dangerous_patterns:
                if col_data.str.contains(pattern, flags=re.IGNORECASE, regex=True).any():
                    raise HTTPException(
                        status_code=422,
                        detail=f"CSV injection pattern detected in column '{col}'"
                    )

    def _validate_structure(self, df: pd.DataFrame):
        """Проверка структуры данных"""
        # Проверка дубликатов колонок
        if len(df.columns) != len(set(df.columns)):
            raise HTTPException(400, "Duplicate column names found")

        missing_cols = [col for col in self.allowed_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )

    def _validate_content(self, df: pd.DataFrame):
        """Основной метод валидации содержимого"""
        self._validate_numeric_columns(df)
        self._check_xss_vectors(df)
        self._detect_sql_injection(df)
        self._validate_data_ranges(df)

    def _validate_numeric_columns(self, df: pd.DataFrame):
        """Проверка числовых колонок"""
        numeric_columns = ['Price']  # Пример числовых колонок

        for col in numeric_columns:
            if col in df.columns:
                try:
                    pd.to_numeric(df[col])
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Column '{col}' must contain numeric values"
                    )

    def _check_xss_vectors(self, df: pd.DataFrame):
        """Поиск XSS-инъекций"""
        xss_patterns = [
            r'<script.*?>',
            r'javascript:',
            r'onerror=',
            r'onload='
        ]

        for col in df.columns:
            col_data = df[col].astype(str)
            for pattern in xss_patterns:
                if col_data.str.contains(pattern, flags=re.IGNORECASE).any():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Potential XSS attack detected in column '{col}'"
                    )

    def _detect_sql_injection(self, df: pd.DataFrame):
        """Поиск SQL-инъекций"""
        sql_keywords = [
            r'\b(SELECT|INSERT|DELETE|DROP|UNION|EXEC|ALTER|CREATE)\b',
            r'--\s',  # Комментарии с пробелом
            r'/\*.*?\*/',  # Блочные комментарии
            r';.*$',  # Команды после точки с запятой
            r'\bWAITFOR\s+DELAY\b'  # Time-based атаки
        ]

        for col in df.columns:
            col_data = df[col].astype(str)
            for keyword in sql_keywords:
                if col_data.str.contains(keyword, flags=re.IGNORECASE).any():
                    raise HTTPException(
                        status_code=400,
                        detail=f"SQL injection pattern detected in column '{col}'"
                    )

    def _validate_data_ranges(self, df: pd.DataFrame):
        """Проверка допустимых диапазонов значений"""
        column_limits = {
            'Price': {'min': 0, 'max': 100000},
            'RAM': {'min': 1, 'max': 128}  # В GB
        }

        for col, limits in column_limits.items():
            if col in df.columns:
                try:
                    values = pd.to_numeric(df[col])
                    if (values < limits['min']).any() or (values > limits['max']).any():
                        raise HTTPException(
                            status_code=400,
                            detail=f"Column '{col}' contains out-of-range values (min: {limits['min']}, max: {limits['max']})"
                        )
                except ValueError:
                    continue

    def validate(self, data: Any) -> None:
        """Реализация абстрактного метода"""
        if isinstance(data, UploadFile):
            raise NotImplementedError("Use validate_upload() for UploadFile objects")
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("DataFrame is empty")
            self._validate_structure(data)
            self._validate_content(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data).__name__}")


class DataSecurityProcessor:
    def __init__(self):
        self.crypto_key = Fernet.generate_key()                         # генерация ключа
        self.cipher = Fernet(self.crypto_key)                           # Инициализируем шифровальщик
        self.salt = b'da_chtob_tbi_zadolbalsya_eto_vzlambivatb'         # Соль для хеширования (должна быть постоянной)
        logger.debug("Initialized DataSecurityProcessor")

    def encrypt_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Шифрование колонки RAM_Size и создание колонки RAM_encrypted"""
        if 'RAM_Size' in df.columns:
            logger.info("Encrypting RAM_Size column")
            df['RAM_encrypted'] = df['RAM_Size'].apply(
                lambda x: self.cipher.encrypt(str(x).encode()).decode()
            )
            logger.debug(f"Encryption samples: {df[['RAM_Size', 'RAM_encrypted']].head().to_dict()}")
        return df

    def decrypt_sample(self, df: pd.DataFrame, n: int = 5) -> dict:
        """Дешифровка и вывод n значений из RAM_encrypted"""
        if 'RAM_encrypted' not in df.columns:
            return {}

        samples = {}
        for val in df['RAM_encrypted'].head(n):
            try:
                decrypted = self.cipher.decrypt(val.encode()).decode()
                samples[val] = decrypted
            except Exception as e:
                samples[val] = f"DECRYPTION_ERROR: {str(e)}"
        return samples

    def _derive_hash_key(self, value: str) -> str:
        """Генерация хеша для значения"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=1000
        )
        return base64.urlsafe_b64encode(kdf.derive(value.encode())).decode()

    def hash_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание хешированной колонки Price_hash"""
        if 'Price' in df.columns:
            df['Price_hash'] = df['Price'].apply(
                lambda x: self._derive_hash_key(str(x))
            )
        return df
