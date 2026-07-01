# =============================================================================
#  features/feature_engineering.py
#
#  FIX for the problems visible in your graphs:
#   1. Per-feature normalization (not global MinMax — that's why spike was huge)
#   2. Separate scalers per source (sensor vs app vs network have different ranges)
#   3. Added temporal + DoS-indicator features
#   4. Robust scaler for outlier-heavy data (RobustScaler instead of MinMax)
# =============================================================================

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler, LabelEncoder
from config import CFG


# Columns the LSTM model will actually train on
MODEL_FEATURE_COLS = [
    # Sensor readings (per-feature scaled)
    "temperature", "humidity", "rssi",
    # Binary flags
    "motion", "relay1", "relay2",
    # Temporal
    "hour", "minute", "day_of_week", "is_night", "is_weekend",
    # Derived behavioural
    "gap_sec",          # inter-event gap (seconds)
    "gap_sec_log",      # log(gap+1)  — compresses large gaps
    "event_rate_1m",    # events in last 60 sec
    "event_rate_5m",    # events in last 300 sec
    "temp_delta",       # temperature change from previous reading
    "temp_roll_mean",   # rolling mean
    "temp_roll_std",    # rolling std
    # Encoded categoricals
    "source_enc",
    "event_enc",
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Merge & sort
# ─────────────────────────────────────────────────────────────────────────────

def merge_sources(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Concat all source DataFrames, sort by timestamp, drop true duplicates."""
    frames = [d for d in dfs if d is not None and not d.empty]
    if not frames:
        raise ValueError("No data provided to merge_sources")
    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["timestamp", "event"])
    print(f"  [merge] {len(merged)} rows from {merged['source'].nunique()} source(s)")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Fill + cast sensor columns
# ─────────────────────────────────────────────────────────────────────────────

def fill_sensor_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sensor columns are NaN for non-sensor rows (TXT/PCAP).
    Strategy:
      - Forward-fill within sensor rows only
      - Then forward-fill globally (so TXT rows inherit last sensor reading)
      - Boolean cols → int
    """
    df = df.copy()
    bool_like = ["motion", "relay1", "relay2"]
    for col in CFG["sensor_features"]:
        if col not in df.columns:
            df[col] = np.nan
    # ffill then bfill to cover leading NaNs
    df[CFG["sensor_features"]] = (
        df[CFG["sensor_features"]]
        .ffill()
        .bfill()
        .fillna(0)
    )
    for c in bool_like:
        if c in df.columns:
            df[c] = df[c].astype(float).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Temporal features
# ─────────────────────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["timestamp"]
    df["hour"]        = ts.dt.hour.astype(float)
    df["minute"]      = ts.dt.minute.astype(float)
    df["day_of_week"] = ts.dt.dayofweek.astype(float)
    df["is_night"]    = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(float)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(float)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Behavioural / temporal delta features
#          THIS is what catches DoS — not raw values but rates and gaps
# ─────────────────────────────────────────────────────────────────────────────

def add_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts_sec = df["timestamp"].astype(np.int64) / 1e9   # unix seconds

    # Inter-event gap
    gap         = ts_sec.diff().fillna(0).clip(0, CFG["gap_sec_cap"])
    df["gap_sec"]     = gap
    df["gap_sec_log"] = np.log1p(gap)

    # Event rate: count events in sliding time windows
    df_indexed = df.set_index("timestamp")
    rate_1m = []
    rate_5m = []
    for t in df["timestamp"]:
        mask_1m = (df["timestamp"] >= t - pd.Timedelta("60s")) & (df["timestamp"] <= t)
        mask_5m = (df["timestamp"] >= t - pd.Timedelta("300s")) & (df["timestamp"] <= t)
        rate_1m.append(mask_1m.sum())
        rate_5m.append(mask_5m.sum())
    df["event_rate_1m"] = np.array(rate_1m, dtype=float)
    df["event_rate_5m"] = np.array(rate_5m, dtype=float)

    # Temperature delta
    if "temperature" in df.columns:
        df["temp_delta"]     = df["temperature"].diff().fillna(0)
        df["temp_roll_mean"] = df["temperature"].rolling(CFG["rolling_window"], min_periods=1).mean()
        df["temp_roll_std"]  = df["temperature"].rolling(CFG["rolling_window"], min_periods=1).std().fillna(0)
    else:
        df["temp_delta"]     = 0.0
        df["temp_roll_mean"] = 0.0
        df["temp_roll_std"]  = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Categorical encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame):
    df         = df.copy()
    le_source  = LabelEncoder()
    le_event   = LabelEncoder()
    df["source_enc"] = le_source.fit_transform(df["source"].astype(str)).astype(float)
    df["event_enc"]  = le_event.fit_transform(df["event"].astype(str)).astype(float)
    return df, le_source, le_event


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — PER-FEATURE RobustScaler
#          FIX: this is what was causing the 91,787 MSE spike —
#          temperature (26–35), rssi (-70 to -80), and boolean (0/1)
#          should NEVER share one MinMaxScaler.
# ─────────────────────────────────────────────────────────────────────────────

def fit_scalers(df: pd.DataFrame, save_path: Path | None = None):
    """
    Fit one RobustScaler per feature column.
    RobustScaler uses median + IQR → robust to the spikes you're seeing.
    Returns: scaled np.ndarray, dict of scalers, list of feature col names
    """
    feature_cols = [c for c in MODEL_FEATURE_COLS if c in df.columns]
    scalers      = {}
    scaled_cols  = []

    for col in feature_cols:
        s = RobustScaler()
        scaled_col = s.fit_transform(df[[col]].values)
        scaled_cols.append(scaled_col)
        scalers[col] = s

    scaled_data = np.hstack(scaled_cols)
    print(f"  [scale] {scaled_data.shape[1]} features scaled with RobustScaler")
    print(f"          Cols: {feature_cols}")

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scalers": scalers, "feature_cols": feature_cols},
                    save_path / "scalers.pkl")
        print(f"  [scale] Scalers saved → {save_path / 'scalers.pkl'}")

    return scaled_data, scalers, feature_cols


def load_scalers(save_path: Path):
    obj = joblib.load(Path(save_path) / "scalers.pkl")
    return obj["scalers"], obj["feature_cols"]


def transform_with_scalers(df: pd.DataFrame, scalers: dict, feature_cols: list) -> np.ndarray:
    """Apply pre-fitted scalers to new data."""
    scaled_cols = []
    for col in feature_cols:
        if col in df.columns and col in scalers:
            scaled_cols.append(scalers[col].transform(df[[col]].values))
        else:
            scaled_cols.append(np.zeros((len(df), 1)))
    return np.hstack(scaled_cols)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Sliding window
# ─────────────────────────────────────────────────────────────────────────────

def create_windows(data: np.ndarray, window: int) -> np.ndarray:
    """
    Shape: (n_windows, window, n_features)
    Stride = 1 (overlapping windows — gives model more training samples).
    """
    n = len(data)
    if n < window:
        raise ValueError(f"Not enough data ({n} rows) for window size {window}. "
                         f"Reduce CFG['window_size'] or add more data.")
    return np.array([data[i: i + window] for i in range(n - window + 1)])


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_pipeline(
    *dfs: pd.DataFrame,
    save_scaler_path: Path | None = None,
    scalers: dict | None = None,
    feature_cols: list | None = None,
) -> tuple:
    """
    Full feature engineering pipeline.
    Returns:
      proc_df        – processed DataFrame (before scaling)
      X_windows      – (n, window, features) array ready for LSTM
      scalers        – fitted scaler dict
      feature_cols   – list of feature column names
      le_source      – LabelEncoder for source
      le_event       – LabelEncoder for event
    """
    print("\n── Feature Engineering ──────────────────────────")
    df = merge_sources(*dfs)
    df = fill_sensor_cols(df)
    df = add_time_features(df)
    print("  [feat] Behavioural features…  (may take a moment for large files)")
    df = add_behavioural_features(df)
    df, le_source, le_event = encode_categoricals(df)

    if scalers is None:
        scaled_data, scalers, feature_cols = fit_scalers(df, save_path=save_scaler_path)
    else:
        scaled_data = transform_with_scalers(df, scalers, feature_cols)

    W         = CFG["window_size"]
    X_windows = create_windows(scaled_data, W)
    print(f"  [wind] Windows: {X_windows.shape}  (n, {W}, {X_windows.shape[2]})")

    return df, X_windows, scaled_data, scalers, feature_cols, le_source, le_event
