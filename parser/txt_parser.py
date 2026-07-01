# =============================================================================
#  parser/txt_parser.py
# =============================================================================

import re
import numpy as np
import pandas as pd
from pathlib import Path
from config import CFG


# Known event-type patterns for richer labelling
_EVENT_PATTERNS = [
    (r"[Mm]otion detected",          "Motion Detected"),
    (r"[Rr]elay\s*2\s*(ON|on)",      "Relay2 ON"),
    (r"[Rr]elay\s*1\s*(ON|on)",      "Relay1 ON"),
    (r"[Ff]an\s+running",            "Fan Running"),
    (r"[Ll]ogin\s+fail",             "Login Failed"),
    (r"[Uu]nauthori[sz]ed",          "Unauthorized Access"),
    (r"[Dd]oor\s+open",              "Door Opened"),
    (r"[Aa]larm",                    "Alarm Triggered"),
]


def classify_event(text: str) -> str:
    for pattern, label in _EVENT_PATTERNS:
        if re.search(pattern, text):
            return label
    return text[:80]   # fallback: first 80 chars of raw event


def parse_txt(filepath: str | Path) -> pd.DataFrame:
    """
    Parse app logs: 'TIMESTAMP - EVENT MESSAGE' per line.
    Returns unified schema.
    """
    filepath = Path(filepath)
    records  = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or " - " not in line:
                continue
            parts = line.split(" - ", 1)
            if len(parts) < 2:
                continue
            try:
                ts    = pd.to_datetime(parts[0].strip(), utc=True)
                raw   = parts[1].strip()
                event = classify_event(raw)
                records.append({
                    "timestamp": ts,
                    "source"  : "application",
                    "event"   : event,
                    "value"   : 1.0,
                    **{c: np.nan for c in CFG["sensor_features"]},
                })
            except Exception:
                continue

    df = pd.DataFrame(records)
    return df if not df.empty else pd.DataFrame(
        columns=["timestamp", "source", "event", "value"] + CFG["sensor_features"]
    )


def parse_txt_dir(dirpath: str | Path) -> pd.DataFrame:
    dirpath = Path(dirpath)
    frames  = [parse_txt(f) for f in sorted(dirpath.glob("*.txt"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
