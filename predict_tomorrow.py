import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from data_preparation import prepare_stock_data


UNIVERSAL_FILES = {
    "xgboost": ("xgboost_universal_nasdaq.json", "xgboost"),
    "lightgbm": ("lgbm_universal_nasdaq.joblib", "joblib"),
    "random_forest": ("rf_universal_nasdaq.joblib", "joblib"),
}


def load_all_universal_models() -> dict:
    """Loads all existing universal models from disk."""
    models = {}
    for key, (filename, mtype) in UNIVERSAL_FILES.items():
        if os.path.exists(filename):
            try:
                if mtype == "xgboost":
                    m = XGBClassifier()
                    m.load_model(filename)
                    models[key] = m
                else:
                    m = joblib.load(filename)
                    models[key] = m
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
    return models


def predict_next_day_multi_model(ticker: str, models: dict):
    """
    Fetches scale-invariant features for the ticker and returns prediction details for all models.
    """
    df = prepare_stock_data(ticker, for_training=False)
    last_row = df.iloc[[-1]].copy()
    last_date = last_row.index[0]
    last_date_str = last_date.strftime('%Y-%m-%d %H:%M') if hasattr(last_date, 'strftime') else str(last_date)

    if 'Target' in last_row.columns:
        last_row.drop(columns=['Target'], inplace=True)

    X_latest_np = np.ascontiguousarray(last_row.to_numpy(), dtype=np.float32)

    results = {}
    buy_count = 0
    sell_count = 0
    confidences = []

    for name, model in models.items():
        if name == "xgboost":
            pred = int(model.predict(X_latest_np)[0])
            proba = float(model.predict_proba(X_latest_np)[0][pred])
        else:
            pred = int(model.predict(last_row)[0])
            proba = float(model.predict_proba(last_row)[0][pred])

        signal = "BUY" if pred == 1 else "SELL"
        conf_pct = proba * 100
        if pred == 1:
            buy_count += 1
        else:
            sell_count += 1
        confidences.append(conf_pct)

        results[name] = {
            "prediction": pred,
            "signal": signal,
            "confidence": conf_pct,
        }

    consensus_signal = "BUY" if buy_count >= sell_count else "SELL"
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return last_date_str, results, consensus_signal, avg_conf


if __name__ == '__main__':
    user_input = input("Enter ticker symbol for universal prediction (e.g., AAPL, NVDA, JPM): ")
    ticker_symbol = user_input.strip().upper()
    
    print(f"\n=== MULTI-MODEL UNIVERSAL INFERENCE: {ticker_symbol} ===")
    
    try:
        models = load_all_universal_models()
        if not models:
            raise FileNotFoundError("No universal model files found. Please run model_training.py first.")

        last_date, model_preds, consensus, avg_conf = predict_next_day_multi_model(ticker_symbol, models)
        
        print("\n" + "=" * 65)
        print("NEXT TRADING DAY PREDICTION RESULTS (MULTI-MODEL ENSEMBLE)")
        print("=" * 65)
        print(f"Target Stock Symbol      : {ticker_symbol}")
        print(f"Last Available Data Date : {last_date}")
        print(f"Overall Ensemble Consensus: {consensus} (Avg Conf: {avg_conf:.2f}%)")
        print("-" * 65)
        print(f"{'MODEL':<18} | {'SIGNAL':<8} | {'CONFIDENCE':<12}")
        print("-" * 65)
        for mname, data in model_preds.items():
            print(f"{mname.upper():<18} | {data['signal']:<8} | {data['confidence']:.2f}%")
        print("=" * 65)
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")

