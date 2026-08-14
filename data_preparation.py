import pandas as pd
import yfinance as yf
import numpy as np
import concurrent.futures


def prepare_stock_data(ticker: str, for_training: bool = True) -> pd.DataFrame:
    """
    Fetches 1 year of hourly OHLCV historical data for a given ticker and builds
    a scale-invariant stationary feature set for universal market model prediction.
    Drops absolute price levels (Open, High, Low, Close, Volume, SMAs) to prevent
    leakage/memorization across tickers.
    
    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        for_training (bool): If True, creates target variable and drops final row.
                             If False, preserves latest hour for live inference.
        
    Returns:
        pd.DataFrame: Scaled, stationary quantitative feature set.
    """
    df = yf.download(ticker, period="1y", interval="1h", auto_adjust=True, progress=False)
    
    if df.empty:
        raise ValueError(f"No data retrieved for ticker '{ticker}'. Please check symbol accuracy or connectivity.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close_series = df['Close']
    
    # 1. Scale-invariant & Stationary Features
    # RSI (14 period)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD Histogram & Normalized MACD
    ema_fast = close_series.ewm(span=12, adjust=False).mean()
    ema_slow = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd_line - signal_line

    # Moving Average Ratios (Relative indicators)
    sma_20 = close_series.rolling(window=20).mean()
    sma_50 = close_series.rolling(window=50).mean()
    df['SMA_20_Ratio'] = close_series / sma_20
    df['SMA_50_Ratio'] = close_series / sma_50

    # Hourly Return and Return Lags (Stationary percentage changes instead of absolute prices)
    df['Hourly_Return'] = close_series.pct_change()
    df['Return_lag_1'] = df['Hourly_Return'].shift(1)
    df['Return_lag_2'] = df['Hourly_Return'].shift(2)
    df['Return_lag_3'] = df['Hourly_Return'].shift(3)

    # 2. Target Variable & Drop Absolute Price Columns
    if for_training:
        df['Target'] = (close_series.shift(-1) > close_series).astype(int)
        
    # Drop absolute price and non-stationary volume columns to prevent memorization
    cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    # 3. Clean NaN values
    df.dropna(inplace=True)

    return df


def _fetch_single_ticker_data(ticker: str) -> tuple:
    """Helper worker function to process a single ticker safely in a thread."""
    try:
        df = prepare_stock_data(ticker, for_training=True)
        return ticker, df, None
    except Exception as e:
        return ticker, None, e


def build_universal_dataset(tickers: list) -> pd.DataFrame:
    """
    Downloads and prepares stock data concurrently across multiple tickers using multithreading,
    then concatenates them into a single universal dataset.
    
    Parameters:
        tickers (list): List of ticker strings (e.g. ['AAPL', 'MSFT', 'NVDA']).
        
    Returns:
        pd.DataFrame: Concatenated dataset across all successful tickers.
    """
    dataset_list = []
    total = len(tickers)
    completed = 0

    print(f"Building Universal Market Dataset concurrently across {total} tickers (max_workers=20)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ticker = {executor.submit(_fetch_single_ticker_data, ticker): ticker for ticker in tickers}
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            completed += 1
            ticker, df, error = future.result()
            
            if error is not None:
                print(f"[{completed}/{total}] Skipping {ticker} due to error: {error}")
            else:
                dataset_list.append(df)
                print(f"[{completed}/{total}] Processed {ticker} successfully ({len(df)} rows).")

    if not dataset_list:
        raise ValueError("Failed to build dataset. No ticker data was successfully processed.")

    universal_df = pd.concat(dataset_list, axis=0, ignore_index=True)
    print(f"\nUniversal Dataset Construction Complete. Total Records: {len(universal_df)}")
    return universal_df


if __name__ == '__main__':
    sample_tickers = ['AAPL', 'MSFT', 'NVDA']
    dataset = build_universal_dataset(sample_tickers)
    print("\nDataset Shape:", dataset.shape)
    print("\nColumns:", dataset.columns.tolist())
    print("\nFirst 5 Records:")
    print(dataset.head())
