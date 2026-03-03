import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

class SpendingAnalyzerAgent:
    def __init__(self):
       
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        
        rf_model_path = os.path.join(base_path, 'models', 'rf_spending_model.pkl')
        iso_model_path = os.path.join(base_path, 'models', 'iso_anomaly_model.pkl')
        vectorizer_path = os.path.join(base_path, 'models', 'vectorizer.pkl')
        
        
        self.rf_model = self._load_model(rf_model_path, "Random Forest spending model")
        self.iso_model = self._load_model(iso_model_path, "Isolation Forest anomaly model")
        self.vectorizer = self._load_model(vectorizer_path, "TF-IDF vectorizer")
    
    def _load_model(self, file_path, model_name):
        """Load a model with robust error handling."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{model_name} file not found: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"{model_name} file is empty: {file_path}")
        
        try:
            with open(file_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load {model_name} from {file_path}: {str(e)}")

    def classify(self, df):
        X_desc = self.vectorizer.transform(df["description"])
        df["predicted_category"] = self.rf_model.predict(X_desc)
        return df

    def detect_anomalies(self, df):
        df["amount_z"] = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-9)

        df["hour"] = df["date"].dt.hour
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["weekday"] = df["date"].dt.weekday

        features = df[["amount", "amount_z", "hour", "day", "month", "weekday"]]

        df["anomaly"] = self.iso_model.predict(features)
        return df

    def summarize(self, df):
        return df.groupby("predicted_category")["amount"].sum().to_dict()

    def process(self, df):
        df = self.classify(df)
        df = self.detect_anomalies(df)
        summary = self.summarize(df)
        return {"detailed": df.to_dict(orient="records"), "summary": summary}