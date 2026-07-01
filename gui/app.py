# =============================================================================
#  gui/app.py  —  Streamlit Investigation Dashboard
#  Run from PROJECT ROOT:  streamlit run gui/app.py
#
#  FIXES:
#   1. ModuleNotFoundError 'config' — ROOT path inserted before all imports
#   2. Groq key read from .env automatically — no need to paste in sidebar
#   3. "No retrain" checkbox — loads saved model so no retraining on restart
#   4. SHAP explainability tab added (IsolationForest feature importance)
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# ── FIX 1: Always resolve ROOT and insert BEFORE any project imports ─────────
ROOT = Path(__file__).resolve().parent.parent   # C:\iot_forensics
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")                       # load GROQ_API_KEY from .env

from config import CFG                           # now always found

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IoT Forensic Investigator", page_icon="🔍", layout="wide")
st.title("🔍 IoT Forensic Investigation Dashboard")
st.caption("LSTM Autoencoder + IsolationForest + SHAP + Groq LLM")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    

    st.subheader("📂 Upload Data Files")
    json_files = st.file_uploader("Sensor JSON logs", type=["json"], accept_multiple_files=True)
    txt_files  = st.file_uploader("App logs (TXT)",   type=["txt"],  accept_multiple_files=True)
    pcap_files = st.file_uploader("Wireshark PCAP",   type=["pcap", "pcapng"], accept_multiple_files=True)

    st.subheader("🎛️ Model Settings")
    window_size     = st.slider("Window size",          10, 60,  CFG["window_size"])
    threshold_sigma = st.slider("Threshold sigma",      1.0, 5.0, CFG["threshold_sigma"], 0.5)
    top_n_explain   = st.slider("LLM explanations (N)", 1, 10, 5)

    # ── FIX 3: Retrain checkbox — uncheck to skip retraining on restart ───────
    retrain = st.checkbox(
        "Retrain model",
        value=True,
        help="Uncheck to load the saved model — much faster on restarts",
    )
    if not retrain:
        st.info("Using saved model from model/saved/")

    CFG["window_size"]     = window_size
    CFG["threshold_sigma"] = threshold_sigma

    run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_meaningful_metrics(timeline: pd.DataFrame) -> dict:
    n = len(timeline)
    lstm_flags = timeline["lstm_flag"].values.astype(int)  if "lstm_flag"  in timeline.columns else np.zeros(n)
    iso_flags  = timeline["iso_flag"].values.astype(int)   if "iso_flag"   in timeline.columns else np.zeros(n)
    comb_flags = timeline["anomaly_flag"].values.astype(int) if "anomaly_flag" in timeline.columns else np.zeros(n)
    both   = int(((lstm_flags == 1) & (iso_flags == 1)).sum())
    either = int(((lstm_flags == 1) | (iso_flags == 1)).sum())
    return {
        "lstm_flagged"    : int(lstm_flags.sum()),
        "iso_flagged"     : int(iso_flags.sum()),
        "combined_flagged": int(comb_flags.sum()),
        "model_agreement" : int(both / max(either, 1) * 100),
        "anomaly_rate"    : float(comb_flags.mean()),
        "total"           : n,
    }


def save_uploads(uploaded, subdir: str):
    dest = ROOT / "data" / subdir
    dest.mkdir(parents=True, exist_ok=True)
    for uf in (uploaded or []):
        (dest / uf.name).write_bytes(uf.read())


def run_shap_analysis(iso_model, scaled_data: np.ndarray, feature_cols: list) -> dict:
    """
    SHAP analysis on IsolationForest.
    Returns dict with shap_values array and feature names.
    """
    try:
        import shap

        explainer   = shap.TreeExplainer(iso_model)
        shap_values = explainer.shap_values(scaled_data)
        mean_abs    = np.abs(shap_values).mean(axis=0)

        # Save PNGs to reports dir so they survive restarts
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
 
        ranked   = sorted(zip(feature_cols, mean_abs), key=lambda x: x[1], reverse=True)
        feats    = [r[0] for r in ranked]
        vals     = [r[1] for r in ranked]
 
        fig, ax = plt.subplots(figsize=(10, max(4, len(feats) * 0.4)))
        ax.barh(feats[::-1], vals[::-1], color="#7F77DD", alpha=0.85)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("SHAP Feature Importance — IsolationForest")
        plt.tight_layout()
        imp_path = CFG["reports_dir"] / "shap_importance.png"
        plt.savefig(imp_path, dpi=130); plt.close()
 
        top_idx = int(np.argmax(mean_abs))
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.scatter(scaled_data[:, top_idx], shap_values[:, top_idx],
                    alpha=0.4, s=12, color="#378ADD")
        ax2.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax2.set_xlabel(f"{feature_cols[top_idx]} (scaled)")
        ax2.set_ylabel("SHAP value")
        ax2.set_title(f"SHAP scatter — {feature_cols[top_idx]}")
        plt.tight_layout()
        scat_path = CFG["reports_dir"] / "shap_scatter.png"
        plt.savefig(scat_path, dpi=130); plt.close()
 
        return {
            "ok"            : True,
            "shap_values"   : shap_values,
            "mean_abs_shap" : mean_abs,
            "feature_cols"  : feature_cols,
            "imp_path"      : str(imp_path),
            "scat_path"     : str(scat_path),
        }
    except ImportError:
        return {"ok": False, "error": "pip install shap"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
 
 
def parse_explanations_from_report(report_path: Path) -> list:
    """Parse LLM explanations back out of the saved .txt report."""
    try:
        text   = report_path.read_text(encoding="utf-8")
        blocks = text.split("\n[#")
        parsed = []
        for block in blocks[1:]:
            lines = block.strip().split("\n")
            header = lines[0]
            try:
                rank_end = header.index("]")
                rank     = int(header[:rank_end])
                ts_str   = header[rank_end+2:].strip()
            except Exception:
                continue
            src = evt = ""
            score_val  = 0.0
            expl_lines = []
            in_expl    = False
            for line in lines[1:]:
                if line.startswith("  Source  :"):
                    src = line.split(":", 1)[1].strip()
                elif line.startswith("  Event   :"):
                    evt = line.split(":", 1)[1].strip()
                elif line.startswith("  Score   :"):
                    try: score_val = float(line.split(":", 1)[1].strip())
                    except: pass
                elif "AI Explanation" in line:
                    in_expl = True
                elif in_expl and line.startswith("    "):
                    expl_lines.append(line.strip())
            if evt:   # only add if we got real content
                parsed.append({
                    "rank": rank, "timestamp": ts_str,
                    "source": src, "event": evt,
                    "combined_score": score_val,
                    "lstm_error": 0.0, "iso_score": 0.0,
                    "explanation": "\n".join(expl_lines),
                })
        return parsed
    except Exception:
        return []
        #return {
            #"shap_values"  : shap_values.values,
            #"mean_abs_shap": mean_abs,
            #"feature_cols" : feature_cols,
            #"ok"           : True,
        #}
    #except ImportError:
        #return {"ok": False, "error": "shap not installed — run: pip install shap"}
    #except Exception as e:
        #return {"ok": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    if not json_files and not txt_files:
        st.error("Please upload at least one JSON or TXT log file.")
        st.stop()

    with st.spinner("Saving uploaded files…"):
        save_uploads(json_files, "json")
        save_uploads(txt_files,  "txt")
        save_uploads(pcap_files, "pcap")

    progress = st.progress(0, text="Ingesting data…")

    from parser.json_parser        import parse_json_dir
    from parser.txt_parser         import parse_txt_dir
    from parser.pcap_features      import parse_pcap_dir
    from features.feature_engineering import run_feature_pipeline
    from model.anomaly_model       import (
        train_lstm, train_isolation_forest,
        load_lstm, load_isolation_forest,
        reconstruction_error, compute_threshold,
        hybrid_anomaly_scores, compute_metrics,
    )
    from timeline.timeline_builder import build_timeline, save_timeline_csv
    from llm.report_generator      import generate_all_explanations, save_text_report

    MODELS  = CFG["model_dir"]
    REPORTS = CFG["reports_dir"]

    # Stage 1
    json_df = parse_json_dir(CFG["json_dir"])
    txt_df  = parse_txt_dir(CFG["txt_dir"])
    pcap_df = parse_pcap_dir(CFG["pcap_dir"])
    progress.progress(15, "Feature engineering…")

    # Stage 2
    (proc_df, X_windows, scaled_data,
     scalers, feature_cols,
     le_source, le_event) = run_feature_pipeline(json_df, txt_df, save_scaler_path=MODELS)

    split   = int(len(X_windows) * 0.80)
    X_train = X_windows[:split]
    progress.progress(30, "Model…")

    # Stage 3 — train or load
    if retrain:
        lstm_model, history = train_lstm(X_train, MODELS)
        iso_model = train_isolation_forest(scaled_data[:int(len(scaled_data)*0.8)], MODELS)
    else:
        try:
            lstm_model = load_lstm(MODELS)
            iso_model  = load_isolation_forest(MODELS)
            history    = None
            st.success(" Loaded saved model — no retraining")
        except Exception as e:
            st.error(f"Could not load saved model: {e}. Enable 'Retrain model'.")
            st.stop()

    progress.progress(60, "Anomaly detection…")

    # Stage 4
    train_errors = reconstruction_error(lstm_model, X_train)
    threshold    = compute_threshold(train_errors)
    scores_df    = hybrid_anomaly_scores(lstm_model, iso_model, X_windows, scaled_data, threshold)
    #metrics      = compute_metrics(scores_df["iso_flag"].values, scores_df["lstm_flag"].values)
    
      

    progress.progress(75, "Timeline…")
    timeline = build_timeline(proc_df, scores_df, pcap_df if not pcap_df.empty else None)
    save_timeline_csv(timeline, REPORTS)
    metrics  = compute_meaningful_metrics(timeline)

    # Stage 5 — SHAP
    progress.progress(82, "SHAP analysis…")
    shap_result = run_shap_analysis(iso_model, scaled_data, feature_cols)

    # Stage 6 — LLM
    progress.progress(88, "LLM explanations…")
    if CFG.get("groq_key"):
        explanations = generate_all_explanations(timeline, top_n=top_n_explain)
    else:
        explanations = []
        st.warning("⚠️ No Groq API key — LLM explanations skipped. Add GROQ_API_KEY to .env")

    # Stage 7 — Report
    report_path = save_text_report(timeline, explanations, metrics, REPORTS)
    progress.progress(100, "Done!")

    # Save everything to session
    st.session_state.update({
        "timeline"    : timeline,
        "scores_df"   : scores_df,
        "explanations": explanations,
        "metrics"     : metrics,
        "history"     : history if retrain else None,
        "threshold"   : threshold,
        "report_path" : report_path,
        "shap_result" : shap_result,
        "iso_model"   : iso_model,
        "scaled_data" : scaled_data,
        "feature_cols": feature_cols,
    })
    st.success(" Pipeline complete!")
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-load existing results on restart (no retraining needed)
# ─────────────────────────────────────────────────────────────────────────────
if "timeline" not in st.session_state:
    csv_path = CFG["reports_dir"] / "timeline.csv"
    if csv_path.exists():
        tl = pd.read_csv(csv_path, parse_dates=["timestamp"])
        tl["timestamp"] = pd.to_datetime(tl["timestamp"], format="mixed", utc=True)
        tl["timestamp_ist"] = tl["timestamp"] + pd.Timedelta(hours=5, minutes=30)
        st.session_state["timeline"] = tl

        # ── Rebuild scores_df from timeline CSV columns ───────
        score_cols = ["lstm_error","iso_score","combined_score",
                      "anomaly_flag","lstm_flag","iso_flag"]
        if all(c in tl.columns for c in score_cols):
            scores_rebuilt = tl[score_cols].copy()
            st.session_state["scores_df"] = scores_rebuilt
            le = scores_rebuilt["lstm_error"]
            st.session_state["threshold"] = float(le.mean() + 3 * le.std())
        elif any(c in tl.columns for c in score_cols):
            scores_rebuilt = pd.DataFrame()
            for c in score_cols:
                scores_rebuilt[c] = tl[c] if c in tl.columns else 0
            st.session_state["scores_df"] = scores_rebuilt
            le = scores_rebuilt["lstm_error"]
            st.session_state["threshold"] = float(le.mean() + 3 * le.std())
 
        # ── Rebuild metrics ───────────────────────────────────
        st.session_state["metrics"] = compute_meaningful_metrics(tl)
 
        # ── Load SHAP PNGs ────────────────────────────────────
        imp_path  = CFG["reports_dir"] / "shap_importance.png"
        scat_path = CFG["reports_dir"] / "shap_scatter.png"
        if imp_path.exists() or scat_path.exists():
            st.session_state["shap_result"] = {
                "ok"       : False,          # no arrays in memory
                "imp_path" : str(imp_path)  if imp_path.exists()  else None,
                "scat_path": str(scat_path) if scat_path.exists() else None,
                "from_disk": True,           # flag: show PNGs not plotly
            }


        # Also try loading latest report text
        reports = sorted(CFG["reports_dir"].glob("forensic_report_*.txt"))
        if reports:
            rp = reports[-1]
            st.session_state["report_path"]   = rp
            parsed = parse_explanations_from_report(rp)
            if parsed:
                st.session_state["explanations"] = parsed

        st.info("📂 Loaded existing results — uncheck 'Retrain model' and click Run to refresh quickly.")

if "timeline" not in st.session_state:
    st.info("👈 Upload your log files and click **Run Pipeline** to begin.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
timeline     = st.session_state["timeline"]
# Convert UTC timestamps to IST (UTC+5:30) for display
timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], utc=True)
timeline["timestamp_ist"] = timeline["timestamp"] + pd.Timedelta(hours=5, minutes=30)
explanations = st.session_state.get("explanations", [])
metrics      = st.session_state.get("metrics", {})
history      = st.session_state.get("history")
threshold    = st.session_state.get("threshold", 0)
shap_result  = st.session_state.get("shap_result", {})

anom_mask = timeline["anomaly_flag"] == 1 if "anomaly_flag" in timeline.columns \
            else pd.Series(False, index=timeline.index)
anom    = timeline[anom_mask]
n_total = len(timeline)
n_anom  = len(anom)

## KPI bar
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total events",    n_total)
c2.metric("Anomalies",       n_anom, f"{n_anom/max(n_total,1)*100:.1f}%")
c3.metric("Sources",         timeline["source"].nunique())
c4.metric("LSTM flagged",    metrics.get("lstm_flagged", "—"))
c5.metric("Model agreement", f"{metrics.get('model_agreement', 0)}%")

# ── 6 Tabs including SHAP ────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "🗂️ Timeline", "📈 Scores",
    "🔬 Events", "🧩 SHAP", "🧠 LLM", "📄 Report"
])

# ── Tab 1: Timeline ──────────────────────────────────────────────────────────
with t1:
    st.subheader("Chronological Event Timeline")
    src_colors = {"sensor": "#378ADD", "application": "#1D9E75", "network": "#BA7517"}
    fig = go.Figure()
    score_col = "combined_score" if "combined_score" in timeline.columns else "value"
    for src, grp in timeline.groupby("source"):
        fig.add_trace(go.Scatter(
            x=grp["timestamp_ist"], y=grp[score_col],
            mode="markers", name=src,
            marker=dict(size=6, color=src_colors.get(src, "#888"), opacity=0.6),
            text=grp["event"], hovertemplate="%{text}<extra>%{x}</extra>",
        ))
    if not anom.empty and score_col in anom.columns:
        fig.add_trace(go.Scatter(
            x=anom["timestamp_ist"], y=anom[score_col],
            mode="markers", name="⚠ Anomaly",
            marker=dict(size=14, color="#E24B4A", symbol="triangle-up"),
            text=anom["event"], hovertemplate="%{text}<extra>ANOMALY</extra>",
        ))
    fig.update_layout(height=400, xaxis_title="Time (IST — UTC+5:30)",
                      yaxis_title="Anomaly Score", hovermode="closest",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Flagged Events")
    if not anom.empty:
        cols = [c for c in ["timestamp_ist","source","event","combined_score","lstm_error","iso_score"] if c in anom.columns]
        st.dataframe(
            anom[cols].sort_values("combined_score", ascending=False)
                       .style.background_gradient(subset=["combined_score"], cmap="Reds"),
            use_container_width=True, height=320,
        )

# ── Tab 2: Scores ────────────────────────────────────────────────────────────
with t2:
    st.subheader("Model Anomaly Scores")
    scores_df = st.session_state.get("scores_df")
    if scores_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=scores_df["lstm_error"], mode="lines",
                                      name="LSTM MAE", line=dict(color="#378ADD", width=1)))
            if threshold:
                fig2.add_hline(y=threshold, line_dash="dash", line_color="#E24B4A",
                               annotation_text=f"Threshold {threshold:.4f}")
            fig2.update_layout(title="LSTM Reconstruction Error (MAE)",
                               yaxis_title="MAE", height=320)
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(y=scores_df["iso_score"], mode="lines",
                                      name="ISO Score", line=dict(color="#1D9E75", width=1)))
            ai = scores_df[scores_df["iso_flag"] == 1].index
            fig3.add_trace(go.Scatter(x=list(ai), y=scores_df.loc[ai,"iso_score"],
                                      mode="markers", name="Anomaly",
                                      marker=dict(color="#E24B4A", size=8)))
            fig3.update_layout(title="IsolationForest Score", height=320)
            st.plotly_chart(fig3, use_container_width=True)
 
        fig4 = px.area(scores_df, y="combined_score",
                       title="Combined Score (LSTM 60% + ISO 40%)",
                       color_discrete_sequence=["#7F77DD"], height=260)
        st.plotly_chart(fig4, use_container_width=True)
 
    # Model stats
        st.subheader("Score Statistics")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("LSTM flagged",  metrics.get("lstm_flagged",  "—"))
        sc2.metric("ISO flagged",   metrics.get("iso_flagged",   "—"))
        sc3.metric("Combined",      metrics.get("combined_flagged","—"))
        sc4.metric("Agreement",     f"{metrics.get('model_agreement',0)}%")

    if history:
        st.subheader("Training Curve")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(y=history["loss"],     name="Train loss"))
        fig5.add_trace(go.Scatter(y=history["val_loss"], name="Val loss",
                                  line=dict(dash="dash")))
        fig5.update_layout(xaxis_title="Epoch", yaxis_title="Huber Loss", height=300)
        st.plotly_chart(fig5, use_container_width=True)

# ── Tab 3: Events ────────────────────────────────────────────────────────────
with t3:
    st.subheader("Event Distribution")
    col1, col2 = st.columns(2)
    with col1:
        if not anom.empty:
            sc = anom["source"].value_counts()
            st.plotly_chart(
                px.bar(sc, title="Anomalies by Source",
                       color=sc.index,
                       color_discrete_map={"sensor":"#378ADD","application":"#1D9E75","network":"#BA7517"}),
                use_container_width=True,
            )
    with col2:
        if not anom.empty:
            te = anom["event"].value_counts().head(10)
            fig_te = px.bar(te, orientation="h", title="Top Anomalous Events")
            fig_te.update_layout(yaxis={"autorange":"reversed"})
            st.plotly_chart(fig_te, use_container_width=True)

    if "hour" in timeline.columns:
        ha   = anom["hour"].value_counts() if not anom.empty else pd.Series(dtype=float)
        ht   = timeline["hour"].value_counts()
        rate = (ha / ht.clip(lower=1) * 100).reindex(range(24), fill_value=0).reset_index()
        rate.columns = ["Hour", "Anomaly %"]
        st.plotly_chart(
            px.bar(rate, x="Hour", y="Anomaly %",
                   title="Anomaly Rate by Hour of Day",
                   color="Anomaly %", color_continuous_scale="Reds"),
            use_container_width=True,
        )

# ── Tab 4: SHAP ──────────────────────────────────────────────────────────────
with t4:
    st.subheader("🧩 SHAP Feature Importance (IsolationForest)")
    st.caption("Shows which features drive anomaly detection the most.")

    if not shap_result:
        st.info("Run the pipeline to generate SHAP analysis.")
 
    elif shap_result.get("from_disk"):
        # ── Loaded from saved PNGs on restart ─────────────────
        st.info("Showing saved SHAP charts from last pipeline run.")
        imp  = shap_result.get("imp_path")
        scat = shap_result.get("scat_path")
        if imp and Path(imp).exists():
            st.image(imp,  caption="Feature Importance — Mean |SHAP|",
                     use_column_width=True)
        if scat and Path(scat).exists():
            st.image(scat, caption="SHAP Scatter — Top Feature",
                     use_column_width=True)
        if not imp and not scat:
            st.warning("PNG files not found. Re-run the pipeline.")
 
    elif not shap_result.get("ok"):
        st.error(f"SHAP error: {shap_result.get('error')}")
        st.code("pip install shap", language="bash")
 
    else:
        # ── Full interactive SHAP (fresh pipeline run) ─────────
        mean_abs    = shap_result["mean_abs_shap"]
        feat_cols   = shap_result["feature_cols"]
        shap_values = shap_result["shap_values"]
 
        shap_df = pd.DataFrame({
            "Feature"    : feat_cols,
            "Mean |SHAP|": mean_abs,
        }).sort_values("Mean |SHAP|", ascending=True)
 
        fig_shap = px.bar(
            shap_df, x="Mean |SHAP|", y="Feature",
            orientation="h",
            title="Feature Importance — Mean Absolute SHAP Value",
            color="Mean |SHAP|", color_continuous_scale="RdYlGn_r",
            height=max(300, len(feat_cols) * 28),
        )
        fig_shap.update_layout(yaxis={"autorange":"reversed"}, showlegend=False)
        st.plotly_chart(fig_shap, use_container_width=True)
 
        top3 = shap_df.sort_values("Mean |SHAP|", ascending=False).head(3)
        st.subheader("Top Contributing Features")
        c1, c2, c3 = st.columns(3)
        for col_w, (_, row) in zip([c1, c2, c3], top3.iterrows()):
            col_w.metric(row["Feature"], f"{row['Mean |SHAP|']:.4f}")
 
        st.subheader("SHAP Scatter — Top Feature")
        top_idx  = int(np.argmax(mean_abs))
        top_name = feat_cols[top_idx]
        scaled_d = st.session_state.get("scaled_data")
        if scaled_d is not None:
            sc_df = pd.DataFrame({
                "Feature value (scaled)": scaled_d[:, top_idx],
                "SHAP value"            : shap_values[:, top_idx],
            })
            fig_sc = px.scatter(sc_df, x="Feature value (scaled)", y="SHAP value",
                                title=f"SHAP scatter — {top_name}",
                                color="SHAP value", color_continuous_scale="RdBu_r",
                                opacity=0.6, height=340)
            fig_sc.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            # show saved PNG as fallback
            imp = shap_result.get("imp_path")
            if imp and Path(imp).exists():
                st.image(imp, use_column_width=True)
 
        st.info("**How to read:** Higher Mean |SHAP| = that feature drives anomaly decisions more. "
                "Positive SHAP value on scatter = that feature value increased the anomaly score.")

# ── Tab 5: LLM ───────────────────────────────────────────────────────────────
with t5:
    st.subheader("🧠 Groq LLM Forensic Explanations")

    # Show key status clearly
    key_ok = bool(CFG.get("groq_key") or os.environ.get("GROQ_API_KEY"))
    if key_ok:
        st.success(f" Groq key active — model: `{CFG.get('llm_model','')}`")
    else:
        st.error(" No Groq API key found. Add `GROQ_API_KEY=gsk_...` to your `.env` file and restart.")

    if not explanations:
        if key_ok:
            st.info("No explanations yet — run the pipeline. "
                    "If you already ran it, the report may not have had anomalies to explain.")
        else:
            st.info("Add your Groq API key and re-run the pipeline.")
    else:
        st.success(f" {len(explanations)} anomalies explained")
        for e in explanations:
            with st.expander(
                f"[#{e['rank']}]  {e['timestamp']}  —  {str(e['event'])[:55]}  "
                f"  (score={e['combined_score']:.4f})",
                expanded=(e["rank"] == 1),
            ):
                #mc1, mc2, mc3 = st.columns(3)
                #mc1.metric("Source",    e.get("source", "—"))
                #mc2.metric("LSTM MAE",  f"{e.get('lstm_error', 0):.5f}")
                #mc3.metric("ISO score", f"{e.get('iso_score',  0):.5f}")
                st.markdown(e.get("explanation", "No explanation text."))

# ── Tab 6: Report ────────────────────────────────────────────────────────────
with t6:
    st.subheader("📄 Forensic Report")
    report_path = st.session_state.get("report_path")

    # Find any existing report even after restart
    if not report_path:
        existing = sorted(CFG["reports_dir"].glob("forensic_report_*.txt"))
        if existing:
            report_path = existing[-1]

    if report_path and Path(report_path).exists():
        report_text = Path(report_path).read_text()
        st.download_button("⬇️ Download Report", data=report_text,
                           file_name=Path(report_path).name, mime="text/plain")
        st.code(report_text, language="text")
    else:
        st.info("Run the pipeline to generate a report.")