# =============================================================================
#  llm/report_generator.py
# =============================================================================

import pandas as pd
from datetime import datetime
from pathlib import Path
from config import CFG
from openai import OpenAI


_CLIENT = None




def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(
            api_key=CFG["groq_api_key"],
            base_url="https://api.groq.com/openai/v1"
        )
    return _CLIENT


SYSTEM_PROMPT = """You are a senior IoT security and forensics analyst.
You receive a window of events from a smart home IoT monitoring system.
Events marked >>> ANOMALY were flagged by an AI model (hybrid LSTM Autoencoder + IsolationForest).

Your analysis must include:
1. **What happened** — plain-English description of the anomalous behaviour
2. **Why it's suspicious** — compare to expected normal patterns
3. **Severity** — LOW / MEDIUM / HIGH with justification
4. **Likely root cause** — sensor fault / DoS attack / intrusion / misconfiguration / physical event
5. **Recommended action** — one concrete next step

Format as bullet points. Be concise (max 180 words). Do not repeat the raw event data."""


def _build_context(timeline: pd.DataFrame, idx: int, context: int) -> str:
    start = max(0, idx - context)
    end   = min(len(timeline), idx + context + 1)
    lines = []
    for _, row in timeline.iloc[start:end].iterrows():
        marker = ">>> ANOMALY" if row.get("anomaly_flag", 0) == 1 else "    normal"
        score  = f"score={row.get('combined_score', 0):.4f}"
        lines.append(
            f"{marker}  [{row['timestamp']}]"
            f"  [{str(row.get('source','')).upper():>12}]"
            f"  {score}"
            f"  {row.get('event', '')}"
        )
    return "\n".join(lines)


def explain_anomaly(
    timeline : pd.DataFrame,
    idx      : int,
    row      : pd.Series,
) -> str:
    context_str = _build_context(timeline, idx, CFG["llm_context_rows"])
    user_msg = (
        f"Anomaly detected:\n"
        f"  Event     : {row.get('event', 'N/A')}\n"
        f"  Source    : {row.get('source', 'N/A')}\n"
        f"  Timestamp : {row.get('timestamp', 'N/A')}\n"
        f"  LSTM error: {row.get('lstm_error', 0):.6f}\n"
        f"  ISO score : {row.get('iso_score', 0):.6f}\n"
        f"  Combined  : {row.get('combined_score', 0):.6f}\n\n"
        f"Event context:\n{context_str}"
    )
    try:
        resp = _get_client().chat.completions.create(
            model=CFG["llm_model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=CFG["llm_max_tokens"]
        )

        return resp.choices[0].message.content
    
    except Exception as e:
        return f"[LLM unavailable: {e}]"


def generate_all_explanations(timeline: pd.DataFrame, top_n: int = 8) -> list[dict]:
    """Explain top-N anomalies by combined_score."""
    anomalies = (
        timeline[timeline["anomaly_flag"] == 1]
        .nlargest(top_n, "combined_score")
    )
    results = []
    for rank, (idx, row) in enumerate(anomalies.iterrows(), 1):
        print(f"\n  Explaining anomaly {rank}/{min(top_n, len(anomalies))}…")
        explanation = explain_anomaly(timeline, idx, row)

        results.append({
            "rank"          : rank,
            "timestamp"     : row["timestamp"],
            "source"        : row.get("source", ""),
            "event"         : row.get("event", ""),
            "combined_score": row.get("combined_score", 0),
            "lstm_error"    : row.get("lstm_error", 0),
            "iso_score"     : row.get("iso_score", 0),
            "explanation"   : explanation,
        })
        print(f"    Done")
    return results


def save_text_report(
    timeline     : pd.DataFrame,
    explanations : list[dict],
    metrics      : dict,
    out_dir      : Path,
):
    out_dir = Path(out_dir)
    path    = out_dir / f"forensic_report_{datetime.now():%Y%m%d_%H%M%S}.txt"
    total   = len(timeline)
    flagged = int(timeline.get("anomaly_flag", pd.Series(0)).sum())

    with open(path, "w") as f:
        _w = f.write

        _w("=" * 72 + "\n")
        _w("   IoT FORENSIC INVESTIGATION REPORT\n")
        _w(f"   Generated : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        _w("=" * 72 + "\n\n")

        _w("EXECUTIVE SUMMARY\n" + "-" * 40 + "\n")
        _w(f"  Total events    : {total}\n")
        _w(f"  Anomalies found : {flagged}  ({flagged/total*100:.1f}%)\n")
        _w(f"  Data sources    : {', '.join(timeline['source'].unique())}\n")
        for k, v in metrics.items():
            _w(f"  {k:<18}: {v}\n")
        _w("\n")

        _w("EVENT DISTRIBUTION\n" + "-" * 40 + "\n")
        for evt, cnt in timeline["event"].value_counts().items():
            _w(f"  {cnt:>5}x  {evt}\n")
        _w("\n")

        _w("TOP ANOMALIES + AI FORENSIC EXPLANATIONS\n" + "-" * 40 + "\n")
        for e in explanations:
            _w(f"\n[#{e['rank']}]  {e['timestamp']}\n")
            _w(f"  Source  : {e['source']}\n")
            _w(f"  Event   : {e['event']}\n")
            _w(f"  Score   : {e['combined_score']:.5f}\n")
            _w(f"\n  AI Explanation:\n")
            for line in e["explanation"].strip().split("\n"):
                _w(f"    {line}\n")
            _w("\n")

        _w("FULL FLAGGED TIMELINE\n" + "-" * 40 + "\n")
        for _, row in timeline[timeline["anomaly_flag"] == 1].iterrows():
            _w(f"  [{row['timestamp']}]  [{str(row.get('source','')).upper():>12}]"
               f"  score={row.get('combined_score',0):.4f}  {row.get('event','')}\n")

    print(f"  Saved → {path}")
    return path
