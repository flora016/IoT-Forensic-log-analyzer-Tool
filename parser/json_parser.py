# =============================================================================
#  parser/json_parser.py
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from config import CFG


def label_sensor_event(row: pd.Series) -> str:
    """Rule-based event labelling from sensor values."""
    if row.get("motion") == True or row.get("motion") == 1:
        return "Motion Detected"
    elif float(row.get("temperature", 0)) > 35:
        return "High Temperature"
    elif row.get("relay2") == True or row.get("relay2") == 1:
        return "Relay2 ON"
    elif row.get("relay1") == True or row.get("relay1") == 1:
        return "Relay1 ON"
    elif float(row.get("rssi", 0)) < -80:
        return "Weak WiFi Signal"
    else:
        return "Normal"


def parse_json(filepath: str | Path) -> pd.DataFrame:
    """
    Parse newline-delimited IoT sensor JSON.
    Returns unified schema: timestamp, source, event, value + sensor cols.
    """
    filepath = Path(filepath)
    df = pd.read_json(filepath, lines=True)

    # ── Timestamp ─────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # ── Boolean cols → int ────────────────────────────────────
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for c in bool_cols:
        df[c] = df[c].astype(int)

    # ── Fill missing sensor cols ──────────────────────────────
    for col in CFG["sensor_features"]:
        if col not in df.columns:
            df[col] = np.nan

    # ── Event label ───────────────────────────────────────────
    df["event"]  = df.apply(label_sensor_event, axis=1)
    df["source"] = "sensor"
    df["value"]  = df["temperature"].astype(float)

    keep = ["timestamp", "source", "event", "value"] + CFG["sensor_features"]
    return df[[c for c in keep if c in df.columns]].copy()


def parse_json_dir(dirpath: str | Path) -> pd.DataFrame:
    """Parse all JSON files in a directory."""
    dirpath = Path(dirpath)
    frames  = [parse_json(f) for f in sorted(dirpath.glob("*.json"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
