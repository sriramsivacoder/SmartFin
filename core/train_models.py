
import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

os.makedirs(MODELS_DIR, exist_ok=True)

SPENDING_MODEL = os.path.join(MODELS_DIR, "svm_spending_model.pkl")
ANOMALY_MODEL = os.path.join(MODELS_DIR, "iso_anomaly_model.pkl")
ARIMA_MODEL = os.path.join(MODELS_DIR, "arima_forecast.pkl")
EMBEDDING_CACHE = os.path.join(MODELS_DIR, "embedding_cache.pkl")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.getsize(path) < 20:
        raise ValueError(f"ERROR: Saved file too small → {path}")



TRAIN_FILE = os.path.join(DATA_DIR, "sample_transactions.csv")

df = pd.read_csv(TRAIN_FILE)

if "description" not in df.columns or "category" not in df.columns:
    raise ValueError("CSV must contain 'description' and 'category' columns!")

print("\n Loaded training data:", len(df), "rows")



print("\n Loading SentenceTransformer model...")
emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print(" Encoding descriptions...")
X = emb_model.encode(df["description"].tolist(), show_progress_bar=True)
y = df["category"].tolist()

save_pickle({"embeddings": X, "labels": y}, EMBEDDING_CACHE)
print(" Embeddings cached:", EMBEDDING_CACHE)



print("\n Training SVM classifier...")
svm = SVC(kernel="linear", probability=True)
svm.fit(X, y)

save_pickle(svm, SPENDING_MODEL)
print(" Saved SVM model:", SPENDING_MODEL)



print("\nPreparing anomaly detection features...")

df["amount_z"] = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-9)


df["hour"] = 12
df["day"] = 15
df["month"] = 6
df["weekday"] = 2

features = df[["amount", "amount_z", "hour", "day", "month", "weekday"]]

print(" Training IsolationForest...")
iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)
iso.fit(features)

save_pickle(iso, ANOMALY_MODEL)
print(" Saved anomaly model:", ANOMALY_MODEL)



print("\n Training LSTM model...")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
monthly_series = df.groupby(df["date"].dt.to_period("M"))["amount"].sum().astype(float)

values = monthly_series.values.reshape(-1, 1)


scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)


X, y = [], []
for i in range(len(scaled_values) - 1):
    X.append(scaled_values[i])
    y.append(scaled_values[i + 1])

X = np.array(X).reshape(-1, 1, 1)
y = np.array(y)


model = Sequential()
model.add(LSTM(32, activation="tanh", input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

model.fit(X, y, epochs=50, batch_size=4, verbose=1)


LSTM_MODEL = os.path.join(MODELS_DIR, "lstm_forecast.h5")
SCALER_FILE = os.path.join(MODELS_DIR, "lstm_scaler.pkl")

model.save(LSTM_MODEL)
save_pickle(scaler, SCALER_FILE)

print("Saved LSTM model:", LSTM_MODEL)
print("Saved scaler:", SCALER_FILE)



print("\n ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")