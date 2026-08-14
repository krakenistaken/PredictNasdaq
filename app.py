import os
import uuid
import threading
import time
import json
import numpy as np
import pandas as pd
import joblib
# pyrefly: ignore [missing-import]
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from xgboost import XGBClassifier

from data_preparation import prepare_stock_data
from model_training import get_all_nasdaq_tickers, train_single_stock_models

app = Flask(__name__)
CORS(app)

# ─── In-memory task store ────────────────────────────────────────────
# { task_id: { "status": "running"|"done"|"error", "progress": int, "stage": str, "message": str, "result": dict|None } }
tasks = {}
tasks_lock = threading.Lock()

# ─── Ticker cache ────────────────────────────────────────────────────
_ticker_cache = {"data": None, "timestamp": 0}
CACHE_TTL = 3600  # 1 hour


def get_cached_tickers():
    """Return cached NASDAQ ticker list, refreshing if stale."""
    now = time.time()
    if _ticker_cache["data"] is None or (now - _ticker_cache["timestamp"]) > CACHE_TTL:
        _ticker_cache["data"] = get_all_nasdaq_tickers()
        _ticker_cache["timestamp"] = now
    return _ticker_cache["data"]


UNIVERSAL_MODELS_CFG = {
    "xgboost": {"file": "xgboost_universal_nasdaq.json", "name": "XGBoost", "icon": "🚀", "type": "xgboost"},
    "lightgbm": {"file": "lgbm_universal_nasdaq.joblib", "name": "LightGBM", "icon": "⚡", "type": "joblib"},
    "random_forest": {"file": "rf_universal_nasdaq.joblib", "name": "Random Forest", "icon": "🌲", "type": "joblib"},
}


# ─── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/tickers")
def api_tickers():
    """Return the full NASDAQ ticker list with metadata."""
    try:
        tickers = get_cached_tickers()
        return jsonify({"status": "ok", "count": len(tickers), "tickers": tickers})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict/<ticker>", methods=["POST"])
def api_predict(ticker):
    """
    1. Immediately return multi-model universal predictions with full context.
    2. Start background task for 4-model ticker-specific training.
    """
    ticker = ticker.strip().upper()
    result = {"ticker": ticker}

    # ── Fetch raw price data for context ──
    try:
        import yfinance as yf
        raw_df = yf.download(ticker, period="5d", interval="1h", auto_adjust=True, progress=False)
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        if not raw_df.empty:
            last_candle = raw_df.iloc[-1]
            last_ts = raw_df.index[-1]
            result["price_info"] = {
                "current_price": round(float(last_candle["Close"]), 2),
                "open": round(float(last_candle["Open"]), 2),
                "high": round(float(last_candle["High"]), 2),
                "low": round(float(last_candle["Low"]), 2),
                "volume": int(last_candle["Volume"]),
                "last_candle_time": last_ts.strftime('%Y-%m-%d %H:%M') if hasattr(last_ts, 'strftime') else str(last_ts),
            }
    except Exception:
        pass

    # ── Phase 1: Universal multi-model prediction (instant) ──
    try:
        df = prepare_stock_data(ticker, for_training=False)
        last_row = df.iloc[[-1]].copy()
        last_date = last_row.index[0]
        last_date_str = last_date.strftime('%Y-%m-%d %H:%M') if hasattr(last_date, 'strftime') else str(last_date)

        feature_values = {}
        for col in last_row.columns:
            feature_values[col] = round(float(last_row[col].iloc[0]), 6)

        if 'Target' in last_row.columns:
            last_row.drop(columns=['Target'], inplace=True)

        X_latest_np = np.ascontiguousarray(last_row.to_numpy(), dtype=np.float32)

        universal_models_res = {}
        buy_count = 0
        sell_count = 0
        confidences = []

        for key, cfg in UNIVERSAL_MODELS_CFG.items():
            file_path = cfg["file"]
            if os.path.exists(file_path):
                try:
                    if cfg["type"] == "xgboost":
                        m = XGBClassifier()
                        m.load_model(file_path)
                        pred = int(m.predict(X_latest_np)[0])
                        proba = float(m.predict_proba(X_latest_np)[0][pred])
                    else:
                        m = joblib.load(file_path)
                        pred = int(m.predict(last_row)[0])
                        proba = float(m.predict_proba(last_row)[0][pred])

                    conf_pct = round(proba * 100, 2)
                    signal = "BUY" if pred == 1 else "SELL"
                    if pred == 1:
                        buy_count += 1
                    else:
                        sell_count += 1
                    confidences.append(conf_pct)

                    universal_models_res[key] = {
                        "name": cfg["name"],
                        "icon": cfg["icon"],
                        "signal": signal,
                        "prediction": pred,
                        "confidence": conf_pct,
                        "model_file": file_path,
                    }
                except Exception as ex:
                    universal_models_res[key] = {
                        "name": cfg["name"],
                        "icon": cfg["icon"],
                        "error": str(ex),
                    }

        if universal_models_res:
            consensus_signal = "BUY" if buy_count >= sell_count else "SELL"
            avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

            # Find XGBoost or first valid model for fallback legacy fields
            primary_model = universal_models_res.get("xgboost") or next(iter(universal_models_res.values()))
            primary_signal = primary_model.get("signal", consensus_signal)
            primary_conf = primary_model.get("confidence", avg_confidence)

            result["universal"] = {
                "signal": primary_signal,
                "confidence": primary_conf,
                "consensus": {
                    "signal": consensus_signal,
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    "total_models": len(confidences),
                    "avg_confidence": avg_confidence,
                },
                "models": universal_models_res,
                "last_date": last_date_str,
                "features": feature_values,
            }
        else:
            result["universal"] = {"error": "Universal models not found. Please run model_training.py first."}

    except Exception as e:
        result["universal"] = {"error": str(e)}

    # ── Phase 2: Start background training for 4 stock-specific models ──
    task_id = str(uuid.uuid4())[:8]

    with tasks_lock:
        tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "stage": "init",
            "message": "Starting multi-model training for " + ticker + "...",
            "result": None,
            "ticker": ticker,
        }

    def background_train():
        def progress_cb(stage, pct, msg):
            with tasks_lock:
                tasks[task_id]["stage"] = stage
                tasks[task_id]["progress"] = pct
                tasks[task_id]["message"] = msg

        try:
            res = train_single_stock_models(ticker, progress_callback=progress_cb)
            with tasks_lock:
                tasks[task_id]["status"] = "done"
                tasks[task_id]["progress"] = 100
                tasks[task_id]["message"] = "Multi-model analysis complete!"
                tasks[task_id]["result"] = res
        except Exception as e:
            with tasks_lock:
                tasks[task_id]["status"] = "error"
                tasks[task_id]["message"] = str(e)

    thread = threading.Thread(target=background_train, daemon=True)
    thread.start()

    result["task_id"] = task_id
    return jsonify({"status": "ok", **result})


@app.route("/api/status/<task_id>")
def api_task_status(task_id):
    """Poll the status of a background training task."""
    with tasks_lock:
        task = tasks.get(task_id)
    if task is None:
        return jsonify({"status": "error", "message": "Task not found"}), 404
    return jsonify(task)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PredictNasdaq - NASDAQ AI Multi-Model Ensemble Engine")
    print("  Starting on http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

