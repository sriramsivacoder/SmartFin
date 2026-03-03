import pickle
import numpy as np
import pandas as pd
import os
try:
    from tensorflow.keras.models import load_model
except ModuleNotFoundError:
    load_model = None
class BudgetOptimizationAgent:

    def __init__(self, sequence_length=3):
       
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.sequence_length = sequence_length
        self.lstm_model = None
        self.scaler = None

        
        if load_model is not None:
            lstm_path = os.path.join(base_path, "models", "lstm_forecast.h5")
            scaler_path = os.path.join(base_path, "models", "lstm_scaler.pkl")

            if os.path.exists(lstm_path) and os.path.exists(scaler_path):
                self.lstm_model = load_model(lstm_path, compile=False)
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

    
    def forecast_next_month(self, monthly_spend: dict):
        values = list(monthly_spend.values())

       
        if self.lstm_model is None or self.scaler is None:
            return float(np.mean(values[-self.sequence_length:]))

        
        if len(values) < self.sequence_length:
            return float(np.mean(values))

        window = values[-self.sequence_length:]
        window_np = np.array(window).reshape(-1, 1)

        
        scaled_window = self.scaler.transform(window_np)

        X = scaled_window.reshape(1, self.sequence_length, 1)

        
        pred_scaled = self.lstm_model.predict(X)[0][0]

       
        pred = self.scaler.inverse_transform([[pred_scaled]])[0][0]

        return float(pred)

    def allocate_budget(self, category_totals: dict, predicted_spend: float):
     
        df_cat = pd.Series(category_totals, dtype=float)
        total_spend = df_cat.sum()

        
        if total_spend == 0:
            return {cat: 0 for cat in df_cat.index}

        
        weights = df_cat / total_spend

        optimized = (weights * predicted_spend).round(2)

        return optimized.to_dict()

    def process(self, monthly_spend: dict, category_totals: dict):
       

        predicted_total = self.forecast_next_month(monthly_spend)
        optimized = self.allocate_budget(category_totals, predicted_total)

        return {
            "next_month_prediction": predicted_total,
            "optimized_budget": optimized
        }