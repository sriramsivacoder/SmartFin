# config/settings.py
import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_PATH, "models")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"

SPENDING_MODEL = os.path.join(MODEL_DIR, "svm_spending_model.pkl")
ANOMALY_MODEL = os.path.join(MODEL_DIR, "iso_anomaly_model.pkl")
EMBEDDING_CACHE = os.path.join(MODEL_DIR, "embeddings_cache.pkl")

ARIMA_MODEL = os.path.join(MODEL_DIR, "arima_forecast.pkl")