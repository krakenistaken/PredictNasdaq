# PredictNasdaq 🚀 https://krakenistaken.github.io/PredictNasdaq/ Page only gives live predictions of universal models(trained with data from every stock in nasdaq), for stock-spesific models, you need to use model locally because it trains models using the live data and it might take 1-5 minutes based on your computing power.

> **NASDAQ AI Multi-Model Ensemble Stock Prediction Engine & Interactive Analytics Platform**

PredictNasdaq is a machine learning platform designed for directional market analysis and automated price trend prediction on NASDAQ instruments. Utilizing a scale-invariant technical feature pipeline, the system combines multiple state-of-the-art ensemble models (**XGBoost**, **LightGBM**, **Random Forest**, and **Support Vector Machines**) to deliver robust consensus signals (**BUY** / **SELL**) with confidence metrics.

---

## 🌟 Key Features

- ⚡ **Instant Universal Multi-Model Ensemble**:
  Instant inference using pre-trained universal market models trained across broad NASDAQ stock data without price memorization.
- 🔄 **Real-Time Async Ticker-Specific Training**:
  Background task pipeline that trains stock-specific ML models on-the-fly for any selected NASDAQ ticker with progress tracking.
- 📊 **Scale-Invariant Feature Engineering**:
  Transforms raw OHLCV price action into stationary technical metrics:
  - **RSI (14-period)** & **MACD Histogram**
  - **Moving Average Ratios** (Price / SMA_20, Price / SMA_50)
  - **Hourly Returns & Return Lags** (Lag 1, Lag 2, Lag 3)
  - **Bollinger Band Relative Position** & **Volatiliy Ratios (ATR)**
- 🌐 **Modern Glassmorphic Web Dashboard**:
  Responsive Flask web interface featuring real-time consensus gauges, live ticker search, raw market context, and individual model confidence breakdowns.
- 💻 **Native Desktop Application**:
  GUI interface built with `CustomTkinter` and `PyWebView` support for desktop execution.
- 💻 **CLI Inference Tool**:
  Lightweight command-line script (`predict_tomorrow.py`) for terminal workflows.

---

## 🏗 Project Architecture

```
PredictNasdaq/
├── app.py                      # Flask Web Application & Async API Endpoints
├── app_desktop.py              # PyWebView Desktop Launcher
├── gui_app.py                  # CustomTkinter GUI Desktop Application
├── launch_desktop.bat          # Windows One-Click Desktop Application Launcher
├── predict_tomorrow.py         # CLI Prediction Script
├── data_preparation.py         # Stationary Feature Extraction Pipeline (yfinance)
├── model_training.py           # Multi-Model Training & Universal Model Generator
├── xgboost_universal_nasdaq.json # Pre-trained Universal XGBoost Model
├── lgbm_universal_nasdaq.joblib  # Pre-trained Universal LightGBM Model
├── rf_universal_nasdaq.joblib    # Pre-trained Universal Random Forest Model
├── svm_universal_nasdaq.joblib   # Pre-trained Universal SVM Model
├── templates/
│   └── index.html              # Modern Web Dashboard Interface
├── static/
│   ├── css/style.css           # Premium Dark Theme Stylesheet
│   └── js/app.js               # Frontend Logic & Live Task Polling
└── requirements.txt            # Python Dependencies
```

---

## 🛠 Tech Stack

- **Core & Data Processing**: Python 3.10+, Pandas, NumPy, yfinance, Joblib
- **Machine Learning & Ensemble**: XGBoost, LightGBM, Scikit-Learn
- **Web Framework & Backend**: Flask, Flask-CORS
- **Frontend & UI**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (ES6+), CustomTkinter, PyWebView

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/krakenistaken/PredictNasdaq.git
cd PredictNasdaq
```

### 2. Create and Activate Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application

### 1. Web Dashboard (Flask)

Start the local web server:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000`.

### 2. Desktop Application

Run the native desktop GUI app:

```bash
python gui_app.py
```

Or on Windows, double-click `launch_desktop.bat`.

### 3. CLI Prediction Tool

Run predictions directly from your terminal:

```bash
python predict_tomorrow.py
```

### 4. Retraining Universal Models

To retrain the universal NASDAQ model suite on freshly fetched historical data:

```bash
python model_training.py
```

---

## 📡 API Reference

### Get Ticker List
- **Endpoint**: `GET /api/tickers`
- **Response**: List of supported NASDAQ ticker symbols and metadata.

### Predict Ticker
- **Endpoint**: `POST /api/predict/<ticker>`
- **Response**: Instant multi-model predictions and background task ID for single-stock training.

### Poll Task Status
- **Endpoint**: `GET /api/status/<task_id>`
- **Response**: Current progress percentage, stage message, and final trained model metrics.

---

## ⚠️ Disclaimer

*PredictNasdaq is developed for educational, analytical, and research purposes only. Stock market trading involves substantial risk of loss. Predictions generated by machine learning models should not be considered financial advice or investment recommendations.*

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
