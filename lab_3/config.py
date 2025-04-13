import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).parent

# Пути к данным
DATA_FILE = BASE_DIR / 'input_data/laptop_price.csv'
MODEL_FILE = BASE_DIR / 'output_data/laptop_price_model.pkl'

# Настройки безопасности
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_COLUMNS = ['Brand',
                   'Processor_Speed',
                   'RAM_Size',
                   'Storage_Capacity',
                   'Screen_Size',
                   'Weight',
                   'Price'
                   ]

# Настройки логирования
LOG_FILE = BASE_DIR / 'app.log'
LOG_LEVEL = 'INFO'
