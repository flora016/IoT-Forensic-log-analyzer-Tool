# =============================================================================
#  main.py  — Full pipeline entry point
#  Run:  python main.py
# =============================================================================

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Make sub-packages importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from config import CFG
from parser.json_parser    import parse_json_dir
from parser.txt_parser     import parse_txt_dir
from parser.pcap_features  import parse_pcap_dir
from features.feature_engineering import run_feature_pipeline
from model.anomaly_model   import (
    train_lstm, train_isolation_forest, load_lstm, load_isolation_forest,
    reconstruction_error, compute_threshold, hybrid_anomaly_scores,
    compute_metrics, plot_training, plot_anomaly_scores,
)
from timeline.timeline_builder import (
    build_timeline, print_timeline, save_timeline_csv, plot_timeline,
)
from llm.report_generator  import generate_all_explanations, save_text_report

REPORTS = CFG["reports_dir"]
MODELS  = CFG["model_dir"]


# ─────────────────────────────────────────────────────────────────────────────
# SHAP analysis
# ─────────────────────────────────────────────────────────────────────────────
 
def run_shap_analysis(iso_model, scaled_data: np.ndarray, feature_cols: list):
    """
    Compute SHAP values for the IsolationForest model and save plots.
    """
    print("\n[SHAP] Computing feature importance…")
    try:
        import shap
 
        explainer   = shap.TreeExplainer(iso_model)
        shap_values = explainer.shap_values(scaled_data)
        mean_abs    = np.abs(shap_values).mean(axis=0)
 
        # ── Print table ───────────────────────────────────────
        print(f"\n  {'Feature':<25} {'Mean |SHAP|':>12}")
        print("  " + "─" * 40)
        ranked = sorted(zip(feature_cols, mean_abs), key=lambda x: x[1], reverse=True)
        for feat, val in ranked:
            print(f"  {feat:<25} {val:>12.5f}")
 
        # ── Bar chart — mean |SHAP| ────────────────────────────
        feats = [r[0] for r in ranked]
        vals  = [r[1] for r in ranked]
 
        fig, ax = plt.subplots(figsize=(10, max(4, len(feats) * 0.4)))
        bars = ax.barh(feats[::-1], vals[::-1], color="#7F77DD", alpha=0.85)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("SHAP Feature Importance — IsolationForest")
        for bar, val in zip(bars, vals[::-1]):
            ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)
        plt.tight_layout()
        path = REPORTS / "shap_importance.png"
        plt.savefig(path, dpi=130)
        plt.close()
        print(f"\n  Saved → {path}")
 
        # ── Scatter: top feature value vs SHAP value ──────────
        top_idx  = int(np.argmax(mean_abs))
        top_feat = feature_cols[top_idx]
 
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.scatter(scaled_data[:, top_idx], shap_values[:, top_idx],
                    alpha=0.4, s=12, color="#378ADD")
        ax2.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax2.set_xlabel(f"{top_feat} (scaled)")
        ax2.set_ylabel("SHAP value")
        ax2.set_title(f"SHAP scatter — {top_feat} (most important feature)")
        plt.tight_layout()
        path2 = REPORTS / "shap_scatter.png"
        plt.savefig(path2, dpi=130)
        plt.close()
        print(f"  Saved → {path2}")
 
        return {"ok": True, "shap_values": shap_values,
                "mean_abs_shap": mean_abs, "feature_cols": feature_cols}
 
    except ImportError:
        print("  [SHAP] skipped — run: pip install shap")
        return {"ok": False}
    except Exception as e:
        print(f"  [SHAP] failed: {e}")
        return {"ok": False}


def run_pipeline(retrain: bool = True):
    print("\n" + "=" * 65)
    print("  IoT FORENSIC INVESTIGATION PIPELINE")
    print("=" * 65)

    # ── STAGE 1: Ingest ──────────────────────────────────────
    print("\n[1/8] DATA INGESTION")
    json_df = parse_json_dir(CFG["json_dir"])
    txt_df  = parse_txt_dir(CFG["txt_dir"])
    pcap_df = parse_pcap_dir(CFG["pcap_dir"])

    if json_df.empty and txt_df.empty:
        print("  ERROR: No JSON or TXT data found. "
              "Place files in data/json/ or data/txt/")
        return

    print(f"  JSON rows : {len(json_df)}")
    print(f"  TXT  rows : {len(txt_df)}")
    print(f"  PCAP bins : {len(pcap_df)}")

    # ── STAGE 2: Feature engineering ───────────────────────
    print("\n[2/8] FEATURE ENGINEERING + NORMALIZATION")
    (proc_df, X_windows, scaled_data,
     scalers, feature_cols,
     le_source, le_event) = run_feature_pipeline(
        json_df, txt_df,
        save_scaler_path=MODELS,
    )

    # Train/test split
    split      = int(len(X_windows) * 0.80)
    X_train    = X_windows[:split]
    X_test     = X_windows[split:]
    print(f"  Train windows : {len(X_train)}   Test windows : {len(X_test)}")

    # ── STAGE 3: Model training ───────────────────────────────
    print("\n[3/8] MODEL TRAINING")

    if retrain:
        lstm_model, history = train_lstm(X_train, MODELS)
        plot_training(history, REPORTS)
        iso_model = train_isolation_forest(scaled_data[:int(len(scaled_data)*0.8)], MODELS)
    else:
        print("  Loading saved models…")
        lstm_model = load_lstm(MODELS)
        iso_model  = load_isolation_forest(MODELS)
        history    = None

    # ── STAGE 4: Anomaly scoring ──────────────────────────────
    print("\n[4/8] ANOMALY DETECTION")
    train_errors = reconstruction_error(lstm_model, X_train)
    threshold    = compute_threshold(train_errors)

    scores_df = hybrid_anomaly_scores(
        lstm_model, iso_model, X_windows, scaled_data, threshold
    )
    plot_anomaly_scores(scores_df, threshold, REPORTS)

    # Accuracy metrics (using ISO flag as pseudo ground truth for LSTM)
    y_true  = scores_df["iso_flag"].values
    y_pred  = scores_df["lstm_flag"].values
    metrics = compute_metrics(y_true, y_pred)

     # ── STAGE 5: SHAP ─────────────────────────────────────────
    print("\n[5/8] SHAP EXPLAINABILITY")
    run_shap_analysis(iso_model, scaled_data, feature_cols)

    # ── STAGE 6: Timeline ─────────────────────────────────────
    print("\n[6/8] TIMELINE RECONSTRUCTION")
    timeline = build_timeline(proc_df, scores_df, pcap_df if not pcap_df.empty else None)
    print_timeline(timeline, n=30)
    save_timeline_csv(timeline, REPORTS)
    plot_timeline(timeline, REPORTS)

    # ── STAGE 7: LLM explanations + report ───────────────────
    print("\n[7/8] LLM FORENSIC EXPLANATIONS")
    if CFG["groq_api_key"]:
        explanations = generate_all_explanations(timeline, top_n=8)
    else:
        print("  [skip] GROQ_API_KEY not set in .env — skipping LLM explanations")
        explanations = []

    print("\n[7/7] SAVING REPORT")
    save_text_report(timeline, explanations, metrics, REPORTS)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Reports saved to: {REPORTS}/")
    print("=" * 65)

    return timeline, scores_df, explanations


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IoT Forensic Pipeline")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Load saved models instead of retraining")
    args = parser.parse_args()
    run_pipeline(retrain=not args.no_retrain)
