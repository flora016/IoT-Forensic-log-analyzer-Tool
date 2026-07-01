import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# ── Load timeline CSV ─────────────────────────────────────────────────────────
# Change this path to your actual reports folder
REPORTS = Path("reports")
timeline = pd.read_csv(REPORTS / "timeline.csv")

# ── Extract flag columns ──────────────────────────────────────────────────────
lstm_flag     = timeline["lstm_flag"].fillna(0).astype(int).values
iso_flag      = timeline["iso_flag"].fillna(0).astype(int).values
combined_flag = timeline["anomaly_flag"].fillna(0).astype(int).values

labels = ["Normal", "Anomaly"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Anomaly Detection — Confusion Matrices", fontsize=15,
             fontweight="bold", y=1.02)

# ── Matrix 1: LSTM vs IsolationForest ─────────────────────────────────────────
# Treat IsoForest as "reference", LSTM as "predicted"
cm1 = confusion_matrix(iso_flag, lstm_flag)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=labels)
disp1.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("LSTM vs IsolationForest\n(ISO = reference, LSTM = predicted)",
                  fontsize=11, pad=12)
axes[0].set_xlabel("LSTM Autoencoder prediction")
axes[0].set_ylabel("IsolationForest (reference)")

# Add percentage annotations
total = cm1.sum()
for i in range(2):
    for j in range(2):
        pct = cm1[i, j] / total * 100
        axes[0].text(j, i + 0.35, f"({pct:.1f}%)",
                     ha="center", va="center", fontsize=9, color="gray")

# ── Matrix 2: LSTM vs Combined flag ───────────────────────────────────────────
cm2 = confusion_matrix(combined_flag, lstm_flag)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=labels)
disp2.plot(ax=axes[1], colorbar=False, cmap="Greens")
axes[1].set_title("LSTM vs Combined Decision\n(Combined = reference, LSTM = predicted)",
                  fontsize=11, pad=12)
axes[1].set_xlabel("LSTM Autoencoder prediction")
axes[1].set_ylabel("Combined flag (reference)")

total2 = cm2.sum()
for i in range(2):
    for j in range(2):
        pct = cm2[i, j] / total2 * 100
        axes[1].text(j, i + 0.35, f"({pct:.1f}%)",
                     ha="center", va="center", fontsize=9, color="gray")

# ── Matrix 3: IsolationForest vs Combined flag ────────────────────────────────
cm3 = confusion_matrix(combined_flag, iso_flag)
disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=labels)
disp3.plot(ax=axes[2], colorbar=False, cmap="Oranges")
axes[2].set_title("IsolationForest vs Combined Decision\n(Combined = reference, ISO = predicted)",
                  fontsize=11, pad=12)
axes[2].set_xlabel("IsolationForest prediction")
axes[2].set_ylabel("Combined flag (reference)")

total3 = cm3.sum()
for i in range(2):
    for j in range(2):
        pct = cm3[i, j] / total3 * 100
        axes[2].text(j, i + 0.35, f"({pct:.1f}%)",
                     ha="center", va="center", fontsize=9, color="gray")

plt.tight_layout()
out_path = REPORTS / "confusion_matrices.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {out_path}")

# ── Print summary stats ───────────────────────────────────────────────────────
print("\n── LSTM vs IsolationForest ─────────────────")
tn, fp, fn, tp = cm1.ravel()
print(f"  True Normal  (both agree normal)   : {tn}")
print(f"  LSTM only flagged (ISO missed)      : {fp}")
print(f"  ISO only flagged  (LSTM missed)     : {fn}")
print(f"  Both flagged (agreement)            : {tp}")
print(f"  Agreement rate                      : {(tn+tp)/(tn+fp+fn+tp)*100:.1f}%")

print("\n── LSTM contribution to Combined ───────────")
tn2, fp2, fn2, tp2 = cm2.ravel()
print(f"  LSTM correctly caught               : {tp2}")
print(f"  LSTM missed (caught by ISO)         : {fn2}")
print(f"  LSTM false positives                : {fp2}")

print("\n── ISO contribution to Combined ────────────")
tn3, fp3, fn3, tp3 = cm3.ravel()
print(f"  ISO correctly caught                : {tp3}")
print(f"  ISO missed (caught by LSTM)         : {fn3}")
print(f"  ISO false positives                 : {fp3}")