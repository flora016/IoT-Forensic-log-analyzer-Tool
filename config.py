# =============================================================================
#  config.py  — Central configuration
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

CFG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    "data_dir"    : BASE_DIR / "data",
    "json_dir"    : BASE_DIR / "data" / "json",
    "txt_dir"     : BASE_DIR / "data" / "txt",
    "pcap_dir"    : BASE_DIR / "data" / "pcap",
    "model_dir"   : BASE_DIR / "model" / "saved",
    "reports_dir" : BASE_DIR / "reports",

    # ── Sensor feature columns (must exist in JSON) ────────────────────────
    "sensor_features": [
        "temperature", "humidity", "motion",
        "relay1", "relay2", "rssi", "uptime"
    ],

    # ── Feature engineering ────────────────────────────────────────────────
    "window_size"    : 20,        # LSTM sequence length
    "rolling_window" : 10,        # rolling stats window
    "gap_sec_cap"    : 300,       # cap inter-event gap at 5 min

    # ── Model ──────────────────────────────────────────────────────────────
    "lstm_units"       : 64,
    "latent_units"     : 32,
    "dropout_rate"     : 0.2,
    "epochs"           : 60,
    "batch_size"       : 16,
    "validation_split" : 0.2,
    "patience"         : 15,      # early stopping

    # ── Anomaly detection ──────────────────────────────────────────────────
    "threshold_sigma"   : 2.2,    # stricter: mean + 3σ  (was 2σ → too sensitive)
    "iso_contamination" : 0.015,   # IsolationForest: expected anomaly fraction

    # ── PCAP features ──────────────────────────────────────────────────────
    "pcap_bin_seconds"  : 5,      # aggregate packets into N-second bins
    "dos_pkt_rate"      : 100,    # packets/sec threshold for DoS flag
    "syn_rate_threshold": 20,     # TCP SYN/sec threshold

    # ── LLM ────────────────────────────────────────────────────────────────
    # Get a free key at: https://console.groq.com
    "groq_api_key"        : os.environ.get("GROQ_API_KEY", ""),
    "llm_model"       : "llama-3.3-70b-versatile",
    "llm_max_tokens"  : 600,
    "llm_context_rows": 15,    # events shown to LLM around each anomaly
}



# Create dirs
for key in ["model_dir", "reports_dir"]:
    CFG[key].mkdir(parents=True, exist_ok=True)
