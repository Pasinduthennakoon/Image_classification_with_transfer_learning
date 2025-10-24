import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Set Kaggle key path
os.environ['KAGGLE_CONFIG_DIR'] = r"C:\Users\ASUS\.kaggle"

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Download dataset to your project folder
api.dataset_download_files('dineshpiyasamara/cats-and-dogs-for-classification', path='./data', unzip=True)

print("✅ Dataset downloaded successfully to ./data")