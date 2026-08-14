import os
import json
import pandas as pd
import numpy as np
import requests
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from data_preparation import prepare_stock_data, build_universal_dataset


def get_all_nasdaq_tickers() -> list:
    """
    Downloads the complete NASDAQ listed stock symbols using the NASDAQ Screener API.
    Falls back to the NASDAQ Trader FTP source if the API is unavailable.
    
    Returns:
        list: A list of dicts with keys: 'symbol', 'name', 'lastsale', 'marketCap', 'sector', 'industry'
    """
    print("Fetching NASDAQ tickers from NASDAQ Screener API...")
    
    api_url = "https://api.nasdaq.com/api/screener/stocks"
    params = {
        "tableonly": "true",
        "exchange": "NASDAQ",
        "limit": 10000,
        "offset": 0,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    
    try:
        resp = requests.get(api_url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        rows = data.get("data", {}).get("table", {}).get("rows", [])
        
        if not rows:
            raise ValueError("API returned empty rows")
        
        tickers = []
        for row in rows:
            symbol = str(row.get("symbol", "")).strip()
            if not symbol or symbol.lower() == "nan":
                continue
            # Skip symbols with special characters (warrants, units, etc.)
            if any(c in symbol for c in ['^', '/', '~']):
                continue
            
            name = str(row.get("name", "")).strip()
            lastsale = str(row.get("lastsale", "")).strip()
            market_cap = str(row.get("marketCap", "")).strip()
            sector = str(row.get("sector", "")).strip()
            industry = str(row.get("industry", "")).strip()
            
            tickers.append({
                "symbol": symbol,
                "name": name,
                "lastsale": lastsale,
                "marketCap": market_cap,
                "sector": sector,
                "industry": industry,
            })
        
        # Deduplicate by symbol
        seen = set()
        unique_tickers = []
        for t in tickers:
            if t["symbol"] not in seen:
                seen.add(t["symbol"])
                unique_tickers.append(t)
        
        print(f"Successfully loaded {len(unique_tickers)} active NASDAQ exchange tickers from API.")
        return unique_tickers
        
    except Exception as e:
        print(f"NASDAQ API failed ({e}), falling back to FTP source...")
        return _get_tickers_from_ftp()


def _get_tickers_from_ftp() -> list:
    """Fallback: fetch tickers from NASDAQ Trader FTP site."""
    url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    df = pd.read_csv(url, sep="|")
    df = df.iloc[:-1].copy()
    
    if 'Test Issue' in df.columns:
        df = df[df['Test Issue'] != 'Y']
    
    df.dropna(subset=['Symbol'], inplace=True)
    raw_tickers = df['Symbol'].tolist()
    
    tickers = []
    seen = set()
    for t in raw_tickers:
        if pd.isna(t):
            continue
        symbol_str = str(t).strip().replace('.', '-').replace('$', '-')
        if symbol_str and symbol_str.lower() != 'nan' and symbol_str not in seen:
            seen.add(symbol_str)
            name = ""
            if 'Security Name' in df.columns:
                match = df[df['Symbol'].astype(str).str.strip() == str(t).strip()]
                if not match.empty:
                    name = str(match.iloc[0].get('Security Name', ''))
            tickers.append({
                "symbol": symbol_str,
                "name": name,
                "lastsale": "",
                "marketCap": "",
                "sector": "",
                "industry": "",
            })
    
    print(f"Successfully loaded {len(tickers)} active NASDAQ tickers from FTP fallback.")
    return tickers


def get_ticker_symbols_only() -> list:
    """Returns just the symbol strings for backward compatibility."""
    ticker_data = get_all_nasdaq_tickers()
    return [t["symbol"] for t in ticker_data]


def train_evaluate_universal_models(df: pd.DataFrame) -> dict:
    """
    Trains and evaluates 3 universal models (XGBoost, LightGBM, Random Forest) on the combined dataset.
    
    Parameters:
        df (pd.DataFrame): Combined scale-invariant feature dataset across all tickers.
        
    Returns:
        dict: Trained model instances.
    """
    X = df.drop(columns=['Target'])
    y = df['Target']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {}

    print("\n" + "=" * 60)
    print(" TRAINING UNIVERSAL MODELS (XGBoost, LightGBM, Random Forest)")
    print("=" * 60)

    # 1. XGBoost
    print("\n[1/3] Training Universal XGBoost Model...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        tree_method='hist',
        device='cuda',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    models['xgboost'] = xgb_model
    xgb_model.save_model("xgboost_universal_nasdaq.json")
    print(" -> Saved xgboost_universal_nasdaq.json")

    # 2. LightGBM
    print("\n[2/3] Training Universal LightGBM Model...")
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        random_state=42,
        verbosity=-1
    )
    lgbm_model.fit(X_train, y_train)
    models['lightgbm'] = lgbm_model
    joblib.dump(lgbm_model, "lgbm_universal_nasdaq.joblib")
    print(" -> Saved lgbm_universal_nasdaq.joblib")

    # 3. Random Forest
    print("\n[3/3] Training Universal Random Forest Model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    joblib.dump(rf_model, "rf_universal_nasdaq.joblib")
    print(" -> Saved rf_universal_nasdaq.joblib")

    # Evaluation summary
    print("\n" + "=" * 60)
    print(" UNIVERSAL MODELS EVALUATION COMPARISON")
    print("=" * 60)
    for key, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        print(f"[{key.upper():<13}] Accuracy: {acc:.4f} | Buy Precision: {prec:.4f}")

    return models


def train_evaluate_universal_xgboost(df: pd.DataFrame) -> XGBClassifier:
    """Backward compatibility wrapper."""
    res = train_evaluate_universal_models(df)
    return res['xgboost']


def train_single_stock_models(ticker: str, progress_callback=None) -> dict:
    """
    Trains 3 dedicated models (XGBoost, LightGBM, Random Forest) for a single stock ticker.
    
    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        progress_callback (callable, optional): Function(stage, percent, message) for progress reporting.
        
    Returns:
        dict: Multi-model evaluation and prediction results.
    """
    def report(stage, pct, msg):
        if progress_callback:
            progress_callback(stage, pct, msg)
        print(f"[{ticker}] [{pct}%] {msg}")
    
    report("data", 10, f"Fetching 1-year hourly historical data for {ticker}...")
    
    # 1. Prepare training data
    train_df = prepare_stock_data(ticker, for_training=True)
    report("data", 25, f"Data prepared: {len(train_df)} records with {len(train_df.columns)} features")
    
    # 2. Split data
    X = train_df.drop(columns=['Target'])
    y = train_df['Target']
    feature_names = X.columns.tolist()
    
    split_idx = int(len(train_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Inference latest row
    inference_df = prepare_stock_data(ticker, for_training=False)
    last_row = inference_df.iloc[[-1]].copy()
    last_date = last_row.index[0]
    last_date_str = last_date.strftime('%Y-%m-%d') if hasattr(last_date, 'strftime') else str(last_date)
    if 'Target' in last_row.columns:
        last_row.drop(columns=['Target'], inplace=True)
    X_latest_df = last_row[feature_names]
    X_latest_np = np.ascontiguousarray(X_latest_df.to_numpy(), dtype=np.float32)

    model_results = {}
    buy_signals = 0
    sell_signals = 0
    confidences = []

    # ---------------- 1. XGBoost ----------------
    report("xgb", 40, "Training XGBoost Model...")
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.02,
        tree_method='hist',
        device='cuda',
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    xgb_model.fit(X_train, y_train)
    xgb_filename = f"xgboost_{ticker}.json"
    xgb_model.save_model(xgb_filename)

    y_pred_xgb = xgb_model.predict(X_test)
    xgb_acc = float(accuracy_score(y_test, y_pred_xgb))
    xgb_prec = float(precision_score(y_test, y_pred_xgb, pos_label=1, zero_division=0))
    xgb_pred = int(xgb_model.predict(X_latest_np)[0])
    xgb_proba = float(xgb_model.predict_proba(X_latest_np)[0][xgb_pred]) * 100

    xgb_fi = [{"feature": fn, "importance": float(imp)} for fn, imp in zip(feature_names, xgb_model.feature_importances_)]
    xgb_fi.sort(key=lambda x: x["importance"], reverse=True)

    if xgb_pred == 1:
        buy_signals += 1
    else:
        sell_signals += 1
    confidences.append(xgb_proba)

    model_results["xgboost"] = {
        "name": "XGBoost",
        "icon": "🚀",
        "model_file": xgb_filename,
        "accuracy": round(xgb_acc, 4),
        "precision_buy": round(xgb_prec, 4),
        "prediction": xgb_pred,
        "signal": "BUY" if xgb_pred == 1 else "SELL",
        "confidence": round(xgb_proba, 2),
        "feature_importance": xgb_fi,
    }

    # ---------------- 2. LightGBM ----------------
    report("lgbm", 70, "Training LightGBM Model...")
    lgbm_model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.02,
        random_state=42,
        verbosity=-1
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_filename = f"lgbm_{ticker}.joblib"
    joblib.dump(lgbm_model, lgbm_filename)

    y_pred_lgbm = lgbm_model.predict(X_test)
    lgbm_acc = float(accuracy_score(y_test, y_pred_lgbm))
    lgbm_prec = float(precision_score(y_test, y_pred_lgbm, pos_label=1, zero_division=0))
    lgbm_pred = int(lgbm_model.predict(X_latest_df)[0])
    lgbm_proba = float(lgbm_model.predict_proba(X_latest_df)[0][lgbm_pred]) * 100

    lgbm_fi = [{"feature": fn, "importance": float(imp)} for fn, imp in zip(feature_names, lgbm_model.feature_importances_)]
    lgbm_fi.sort(key=lambda x: x["importance"], reverse=True)

    if lgbm_pred == 1:
        buy_signals += 1
    else:
        sell_signals += 1
    confidences.append(lgbm_proba)

    model_results["lightgbm"] = {
        "name": "LightGBM",
        "icon": "⚡",
        "model_file": lgbm_filename,
        "accuracy": round(lgbm_acc, 4),
        "precision_buy": round(lgbm_prec, 4),
        "prediction": lgbm_pred,
        "signal": "BUY" if lgbm_pred == 1 else "SELL",
        "confidence": round(lgbm_proba, 2),
        "feature_importance": lgbm_fi,
    }

    # ---------------- 3. Random Forest ----------------
    report("rf", 90, "Training Random Forest Model...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_filename = f"rf_{ticker}.joblib"
    joblib.dump(rf_model, rf_filename)

    y_pred_rf = rf_model.predict(X_test)
    rf_acc = float(accuracy_score(y_test, y_pred_rf))
    rf_prec = float(precision_score(y_test, y_pred_rf, pos_label=1, zero_division=0))
    rf_pred = int(rf_model.predict(X_latest_df)[0])
    rf_proba = float(rf_model.predict_proba(X_latest_df)[0][rf_pred]) * 100

    rf_fi = [{"feature": fn, "importance": float(imp)} for fn, imp in zip(feature_names, rf_model.feature_importances_)]
    rf_fi.sort(key=lambda x: x["importance"], reverse=True)

    if rf_pred == 1:
        buy_signals += 1
    else:
        sell_signals += 1
    confidences.append(rf_proba)

    model_results["random_forest"] = {
        "name": "Random Forest",
        "icon": "🌲",
        "model_file": rf_filename,
        "accuracy": round(rf_acc, 4),
        "precision_buy": round(rf_prec, 4),
        "prediction": rf_pred,
        "signal": "BUY" if rf_pred == 1 else "SELL",
        "confidence": round(rf_proba, 2),
        "feature_importance": rf_fi,
    }

    report("done", 100, "All 3 models trained and evaluated successfully!")

    consensus_signal = "BUY" if buy_signals >= sell_signals else "SELL"
    avg_confidence = round(sum(confidences) / len(confidences), 2)

    return {
        "ticker": ticker,
        "total_records": len(train_df),
        "last_date": last_date_str,
        "consensus": {
            "signal": consensus_signal,
            "buy_count": buy_signals,
            "sell_count": sell_signals,
            "avg_confidence": avg_confidence,
        },
        "models": model_results,
        # Backwards compatibility fields mapping to primary XGBoost model
        "accuracy": xgb_acc,
        "precision_buy": xgb_prec,
        "prediction": xgb_pred,
        "signal": "BUY" if xgb_pred == 1 else "SELL",
        "confidence": round(xgb_proba, 2),
        "last_date": last_date_str,
        "model_file": xgb_filename,
        "feature_importance": xgb_fi,
    }


def train_single_stock_model(ticker: str, progress_callback=None) -> dict:
    """Backward compatibility wrapper."""
    return train_single_stock_models(ticker, progress_callback=progress_callback)


if __name__ == '__main__':
    # 1. Fetch full NASDAQ exchange ticker directory
    tickers = get_ticker_symbols_only()
    
    # 2. Build multi-ticker dataset
    universal_df = build_universal_dataset(tickers)
    
    # 3. Train all 3 universal models
    universal_models = train_evaluate_universal_models(universal_df)
    
    print("\n[SUCCESS] Universal Market Models (XGBoost, LightGBM, Random Forest) trained & saved successfully.")

