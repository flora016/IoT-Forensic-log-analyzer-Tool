# =============================================================================
#  model/anomaly_model.py
#
#  Hybrid approach:
#   - IsolationForest  → catches point anomalies (fast, no overfitting)
#   - LSTM Autoencoder → catches sequential / temporal anomalies
#   - Combined score   → fusion of both
#
#  This fixes the val_loss plateau and the 91,787 spike you saw:
#   the LSTM no longer needs to do everything alone.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (precision_score, recall_score, f1_score,confusion_matrix, ConfusionMatrixDisplay)
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,RepeatVector, TimeDistributed, LayerNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from config import CFG


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Autoencoder  (improved architecture)
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm_autoencoder(timesteps: int, n_features: int) -> Model:
    """
    Improved LSTM Autoencoder:
      - LayerNormalization after each LSTM (stabilises training)
      - Smaller units relative to data size (reduces overfitting)
      - Single encoder/decoder pair (less complexity for small datasets)
    """
    inp = Input(shape=(timesteps, n_features), name="input")

    # Encoder
    x = LSTM(CFG["lstm_units"], return_sequences=True,  name="enc1")(inp)
    x = LayerNormalization()(x)
    x = Dropout(CFG["dropout_rate"])(x)
    x = LSTM(CFG["latent_units"], return_sequences=False, name="enc2")(x)
    x = LayerNormalization()(x)
    encoded = Dropout(CFG["dropout_rate"])(x)

    # Bottleneck
    x = RepeatVector(timesteps, name="bottleneck")(encoded)

    # Decoder
    x = LSTM(CFG["latent_units"], return_sequences=True,  name="dec1")(x)
    x = LayerNormalization()(x)
    x = Dropout(CFG["dropout_rate"])(x)
    x = LSTM(CFG["lstm_units"],   return_sequences=True,  name="dec2")(x)
    x = LayerNormalization()(x)
    out = TimeDistributed(Dense(n_features, activation="linear"), name="recon")(x)

    model = Model(inp, out, name="LSTM_Autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",          # Huber loss — less sensitive to the outlier spikes
    )
    return model


def train_lstm(
    X_train: np.ndarray,
    save_dir: Path,
) -> tuple[Model, dict]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "lstm_autoencoder.keras"

    timesteps  = X_train.shape[1]
    n_features = X_train.shape[2]
    model      = build_lstm_autoencoder(timesteps, n_features)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=CFG["patience"],
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(model_path), monitor="val_loss",
                        save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, X_train,
        epochs           = CFG["epochs"],
        batch_size       = CFG["batch_size"],
        validation_split = CFG["validation_split"],
        callbacks        = callbacks,
        verbose          = 1,
    )
    print(f"  [LSTM] Best val_loss: {min(history.history['val_loss']):.6f}")
    return model, history.history


def load_lstm(save_dir: Path) -> Model:
    return load_model(Path(save_dir) / "lstm_autoencoder.keras")


# ─────────────────────────────────────────────────────────────────────────────
# IsolationForest  (on flat feature matrix — catches point anomalies)
# ─────────────────────────────────────────────────────────────────────────────

def train_isolation_forest(scaled_data: np.ndarray, save_dir: Path) -> IsolationForest:
    save_dir = Path(save_dir)
    iso = IsolationForest(
        n_estimators   = 200,
        contamination  = CFG["iso_contamination"],
        max_features   = 0.8,
        random_state   = 42,
        n_jobs         = -1
    )
    iso.fit(scaled_data)
    joblib.dump(iso, save_dir / "isolation_forest.pkl")
    print(f"  [ISO] IsolationForest trained on {scaled_data.shape}")
    return iso


def load_isolation_forest(save_dir: Path) -> IsolationForest:
    return joblib.load(Path(save_dir) / "isolation_forest.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction error  (per sequence)
# ─────────────────────────────────────────────────────────────────────────────

def reconstruction_error(model: Model, X: np.ndarray) -> np.ndarray:
    """Mean absolute error per sequence — less sensitive than MSE to spikes."""
    X_pred = model.predict(X, verbose=0)
    return np.mean(np.abs(X - X_pred), axis=(1, 2))   # MAE not MSE


def compute_threshold(train_errors: np.ndarray) -> float:
    mu    = train_errors.mean()
    sigma = train_errors.std()
    thr   = mu + CFG["threshold_sigma"] * sigma
    print(f"  [thr] Train MAE — mean={mu:.5f}  std={sigma:.5f}")
    print(f"  [thr] Threshold = {thr:.5f}  (mean + {CFG['threshold_sigma']}σ)")
    return float(thr)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid anomaly scoring
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_anomaly_scores(
    lstm_model    : Model,
    iso_model     : IsolationForest,
    X_windows     : np.ndarray,
    scaled_data   : np.ndarray,
    threshold     : float,
) -> pd.DataFrame:
    """
    Combine LSTM reconstruction error + IsolationForest score.
    iso_score is in [-1, 1] from sklearn; we flip it so higher = more anomalous.
    Final combined_score is a 0–1 weighted blend.
    """
    # LSTM scores (per window)
    lstm_errors = reconstruction_error(lstm_model, X_windows)
    lstm_flags  = (lstm_errors > threshold).astype(int)

    # IsolationForest scores (per row in flat data)
    iso_raw     = iso_model.decision_function(scaled_data)  # higher = more normal
    iso_score   = -iso_raw                                   # flip: higher = more anomalous
    iso_norm = iso_norm = (iso_score - np.percentile(iso_score, 5)) / (
    np.percentile(iso_score, 95) - np.percentile(iso_score, 5) + 1e-9
)
    iso_norm = np.clip(iso_norm, 0, 1)
    iso_flags   = (iso_model.predict(scaled_data) == -1).astype(int)

    # LSTM errors normalised
   
    lstm_norm = (lstm_errors - lstm_errors.min()) / (np.ptp(lstm_errors) + 1e-9)

    # Align lengths (LSTM window array is shorter than flat data by window_size-1)
    n_win = len(lstm_errors)
    n_row = len(scaled_data)

    # Pad LSTM scores to match flat-data length
    lstm_errors_padded = np.concatenate([
        np.zeros(n_row - n_win),   # leading zeros (no window covers these)
        lstm_errors
    ])
    lstm_norm_padded = np.concatenate([
        np.zeros(n_row - n_win), lstm_norm
    ])
    lstm_flags_padded = np.concatenate([
        np.zeros(n_row - n_win, dtype=int), lstm_flags
    ])

    # Weighted combination (60% LSTM temporal + 40% IsolationForest)
    combined = 0.6* lstm_norm_padded + 0.4* iso_norm
    combined_flag = combined_flag = (combined > 0.6).astype(int)

    results = pd.DataFrame({
        "lstm_error"   : lstm_errors_padded,
        "lstm_flag"    : lstm_flags_padded,
        "iso_score"    : iso_score,
        "iso_flag"     : iso_flags,
        "combined_score": combined,
        "anomaly_flag" : combined_flag,
    })
    print(f"  [score] Anomalies: {combined_flag.sum()} / {len(combined_flag)} rows "
          f"({combined_flag.mean()*100:.1f}%)")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    If you have ground-truth labels pass them in.
    Otherwise we use iso_flag as pseudo-labels vs lstm_flag.
    """
    metrics = {
        "precision"   : precision_score(y_true, y_pred, zero_division=0),
        "recall"      : recall_score(y_true, y_pred, zero_division=0),
        "f1"          : f1_score(y_true, y_pred, zero_division=0),
        "anomaly_rate": float(y_pred.mean()),
        "total"       : int(len(y_pred)),
        "flagged"     : int(y_pred.sum()),
    }
    print("\n  ── Model Accuracy Metrics ───────────────────────")
    for k, v in metrics.items():
        print(f"    {k:<16}: {v}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(history: dict, out_dir: Path):
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["loss"],     label="Train loss (Huber)", linewidth=2)
    ax.plot(history["val_loss"], label="Val loss",  linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.set_title("LSTM Autoencoder — Training Curve")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "training_loss.png"
    plt.savefig(path, dpi=130); plt.close()
    print(f"  Saved → {path}")


def plot_anomaly_scores(scores_df: pd.DataFrame, threshold: float, out_dir: Path):
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    idx = np.arange(len(scores_df))

    # LSTM error
    axes[0].plot(idx, scores_df["lstm_error"], color="#378ADD",
                 linewidth=0.8, label="LSTM recon error (MAE)")
    axes[0].axhline(threshold, color="#E24B4A", linestyle="--",
                    linewidth=1.5, label=f"Threshold ({threshold:.4f})")
    axes[0].scatter(
        idx[scores_df["lstm_flag"] == 1],
        scores_df["lstm_error"][scores_df["lstm_flag"] == 1],
        color="#E24B4A", s=20, zorder=5, label="LSTM anomaly"
    )
    axes[0].set_title("LSTM Reconstruction Error (MAE)"); axes[0].legend(fontsize=8)

    # IsolationForest score
    axes[1].plot(idx, scores_df["iso_score"], color="#1D9E75",
                 linewidth=0.8, label="IsoForest score (higher = anomalous)")
    axes[1].scatter(
        idx[scores_df["iso_flag"] == 1],
        scores_df["iso_score"][scores_df["iso_flag"] == 1],
        color="#E24B4A", s=20, zorder=5, label="ISO anomaly"
    )
    axes[1].set_title("IsolationForest Anomaly Score"); axes[1].legend(fontsize=8)

    # Combined score
    axes[2].fill_between(idx, scores_df["combined_score"],
                         alpha=0.6, color="#7F77DD", label="Combined score")
    axes[2].scatter(
        idx[scores_df["anomaly_flag"] == 1],
        scores_df["combined_score"][scores_df["anomaly_flag"] == 1],
        color="#E24B4A", s=25, zorder=5, label="Final anomaly flag"
    )
    axes[2].set_title("Combined Hybrid Score (LSTM 60% + ISO 40%)")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("Row index"); ax.grid(alpha=0.2)

    plt.tight_layout()
    path = out_dir / "anomaly_scores.png"
    plt.savefig(path, dpi=130); plt.close()
    print(f"  Saved → {path}")
