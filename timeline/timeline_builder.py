# =============================================================================
#  timeline/timeline_builder.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from config import CFG


def build_timeline(
    proc_df    : pd.DataFrame,
    scores_df  : pd.DataFrame,
    pcap_df    : pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Attach anomaly scores back to the original event rows.
    Optionally merge PCAP network feature rows into the timeline.
    """
    timeline = proc_df.copy()
    n        = len(timeline)

    # Align scores (scores_df may have same length as proc_df)
    for col in ["lstm_error", "iso_score", "combined_score",
                "anomaly_flag", "lstm_flag", "iso_flag"]:
        if col in scores_df.columns:
            arr = scores_df[col].values
            # Pad/trim to match proc_df length
            if len(arr) < n:
                arr = np.concatenate([np.zeros(n - len(arr)), arr])
            timeline[col] = arr[:n]

    # Merge PCAP network rows (dos_flag, pkt_rate, etc.)
    if pcap_df is not None and not pcap_df.empty:
        net_cols = ["timestamp", "source", "event", "value",
                    "pkt_rate", "bytes_rate", "tcp_syn_rate",
                    "dos_flag", "syn_flag", "proto_entropy"]
        net_cols = [c for c in net_cols if c in pcap_df.columns]
        net_df   = pcap_df[net_cols].copy()
        net_df["anomaly_flag"] = net_df.get("dos_flag", pd.Series(0, index=net_df.index))
        net_df["combined_score"] = net_df.get("pkt_rate", 0) / (CFG["dos_pkt_rate"] * 2)
        net_df["lstm_error"] = (net_df["pkt_rate"] / CFG["dos_pkt_rate"]).clip(0, 1) if "pkt_rate" in net_df.columns else 0.0
        net_df["iso_score"]  = net_df["dos_flag"].astype(float) if "dos_flag" in net_df.columns else 0.0

        timeline = pd.concat([timeline, net_df], ignore_index=True)
        timeline = timeline.sort_values("timestamp").reset_index(drop=True)
        print(f"  [timeline] Merged {len(net_df)} PCAP rows")

    print(f"  [timeline] {len(timeline)} total events | "
          f"{int(timeline['anomaly_flag'].sum())} flagged")
    return timeline


def print_timeline(timeline: pd.DataFrame, n: int = 40):
    """Pretty-print the last N events of the timeline."""
    print(f"\n{'─'*110}")
    print(f"  {'TIMESTAMP':<28} {'SOURCE':<14} {'FLAG':<12} {'SCORE':>8}  EVENT")
    print(f"{'─'*110}")
    for _, row in timeline.tail(n).iterrows():
        flag = "⚠  ANOMALY" if row.get("anomaly_flag", 0) == 1 else "   normal"
        score = f"{row.get('combined_score', 0):.4f}"
        ts  = str(row["timestamp"])[:25]
        src = str(row.get("source", ""))[:12]
        evt = str(row.get("event", ""))[:55]
        print(f"  {ts:<28} {src:<14} {flag:<12} {score:>8}  {evt}")
    print(f"{'─'*110}")


def save_timeline_csv(timeline: pd.DataFrame, out_dir: Path):
    out_dir = Path(out_dir)
    path    = out_dir / "timeline.csv"
    cols    = ["timestamp", "source", "event", "anomaly_flag",
               "combined_score", "lstm_error", "iso_score",
               "lstm_flag", "iso_flag"]          # ← add these two
    cols    = [c for c in cols if c in timeline.columns]
    timeline[cols].to_csv(path, index=False)
    print(f"  Saved → {path}")


def plot_timeline(timeline: pd.DataFrame, out_dir: Path):
    """4-panel investigation dashboard."""
    out_dir = Path(out_dir)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("IoT Forensic — Timeline & Anomaly Dashboard",
                 fontsize=15, fontweight="bold")

    src_colors = {"sensor": "#378ADD", "application": "#1D9E75",
                  "network": "#BA7517", "unknown": "#888780"}

    anom = timeline[timeline["anomaly_flag"] == 1]

    # ── 1. Timeline ───────────────────────────────────────────
    ax1 = fig.add_subplot(3, 2, (1, 2))
    for src, grp in timeline.groupby("source"):
        ax1.scatter(grp["timestamp"], grp.get("combined_score", 0),
                    s=8, alpha=0.45, label=src,
                    color=src_colors.get(src, "#888780"))
    if not anom.empty:
        ax1.scatter(anom["timestamp"], anom.get("combined_score", 0),
                    s=70, color="#E24B4A", zorder=6,
                    marker="^", label="Anomaly")
    ax1.set_title("Event Timeline — Combined Anomaly Score")
    ax1.set_ylabel("Score"); ax1.legend(fontsize=8, ncol=4)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha="right")

    # ── 2. Anomalies by source ────────────────────────────────
    ax2 = fig.add_subplot(3, 2, 3)
    if not anom.empty:
        sc = anom["source"].value_counts()
        ax2.bar(sc.index, sc.values,
                color=[src_colors.get(s, "#888780") for s in sc.index])
        for i, (s, v) in enumerate(sc.items()):
            ax2.text(i, v + 0.2, str(v), ha="center", fontsize=10)
    ax2.set_title("Anomalies by Source"); ax2.set_ylabel("Count")

    # ── 3. Top anomalous events ───────────────────────────────
    ax3 = fig.add_subplot(3, 2, 4)
    if not anom.empty:
        te = anom["event"].value_counts().head(8)
        ax3.barh(range(len(te)), te.values, color="#E24B4A", alpha=0.8)
        ax3.set_yticks(range(len(te)))
        ax3.set_yticklabels([e[:42] for e in te.index], fontsize=8)
        ax3.invert_yaxis()
    ax3.set_title("Top Flagged Events"); ax3.set_xlabel("Count")

    # ── 4. Hour-of-day heatmap ────────────────────────────────
    ax4 = fig.add_subplot(3, 2, 5)
    if "hour" in timeline.columns:
        ha = timeline[timeline["anomaly_flag"] == 1]["hour"].value_counts()
        ht = timeline["hour"].value_counts()
        rate = (ha / ht.clip(lower=1) * 100).reindex(range(24), fill_value=0)
        ax4.bar(rate.index, rate.values, color="#7F77DD", alpha=0.85)
        ax4.axvspan(22, 24, alpha=0.08, color="red")
        ax4.axvspan(0, 6, alpha=0.08, color="red")
        ax4.set_title("Anomaly Rate by Hour of Day (%)")
        ax4.set_xlabel("Hour"); ax4.set_ylabel("%")
        ax4.set_xticks(range(0, 24, 2))

    # ── 5. Score distribution ─────────────────────────────────
    ax5 = fig.add_subplot(3, 2, 6)
    norm = timeline[timeline["anomaly_flag"] == 0]["combined_score"]
    anm  = timeline[timeline["anomaly_flag"] == 1]["combined_score"]
    if not norm.empty:
        ax5.hist(norm, bins=30, alpha=0.7, color="#378ADD", label="Normal")
    if not anm.empty:
        ax5.hist(anm,  bins=30, alpha=0.7, color="#E24B4A", label="Anomaly")
    ax5.set_title("Score Distribution"); ax5.legend(fontsize=8)
    ax5.set_xlabel("Combined Score")

    plt.tight_layout()
    path = out_dir / "timeline_dashboard.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")
