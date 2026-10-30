import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- config
# sensor: (mean, std, clip_min, clip_max, ideal_lo, ideal_hi, penalty_below, penalty_above, weight)
SENSORS = {
    "Temperature": (70, 10, 30, 120, 50, 75, 2, 4, 0.20),
    "Vibration":   (4, 1.5, 0, 10, 0, 0, 0, 10, 0.30),
    "Pressure":    (100, 8, 50, 150, 100, 100, 5, 5, 0.15),
    "RPM":         (1500, 120, 500, 2000, 1400, 1600, 0.5, 0.5, 0.15),
    "Current":     (30, 5, 5, 60, 20, 40, 4, 4, 0.20),
}
LABELS = ["Critical", "Warning", "Healthy"]
COND_COLORS = {"Healthy": "#009E73", "Warning": "#E69F00", "Critical": "#D55E00"}
SENSOR_COLORS = dict(zip(SENSORS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]))
WIN = 25  # moving-average window

# ---------------------------------------------------------------- data
rng = np.random.default_rng(42)
N = 1000
data = pd.DataFrame({
    name: rng.normal(m, s, N).clip(lo, hi)
    for name, (m, s, lo, hi, *_) in SENSORS.items()
})

# ---------------------------------------------------------------- health score
def band_score(x, lo, hi, k_low, k_high):
    """100 inside the ideal band [lo, hi], decreasing linearly outside it."""
    penalty = k_low * (lo - x).clip(lower=0) + k_high * (x - hi).clip(lower=0)
    return (100 - penalty).clip(lower=0)

scores = pd.DataFrame({n: band_score(data[n], *p[4:8]) for n, p in SENSORS.items()})
data["Health_Score"] = sum(scores[n] * p[8] for n, p in SENSORS.items())
data["Condition"] = pd.cut(
    data["Health_Score"], [-np.inf, 60, 80, np.inf], labels=LABELS, right=False
)

# ---------------------------------------------------------------- model
features = list(SENSORS)
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data["Condition"], test_size=0.2, random_state=42, stratify=data["Condition"]
)
model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
pred = model.predict(X_test)
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

print(f"Model accuracy: {accuracy_score(y_test, pred):.2%}\n")
print(classification_report(y_test, pred, zero_division=0))
print("Sensor importance:\n", importance.round(3))

latest = data.iloc[-1]
print(f"\nLatest reading -> Health score: {latest.Health_Score:.1f}/100, Condition: {latest.Condition}")

# ---------------------------------------------------------------- figure (paper style, line graphs)
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

def smooth(df):
    return df.rolling(WIN, center=True).mean()

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
(a, b), (c, d) = axes

# (a) overall health score
score = data["Health_Score"]
a.plot(score, color="0.65", lw=0.5, label="Raw")
a.plot(smooth(score), color="#0072B2", lw=1.4, label=f"{WIN}-pt moving avg.")
a.axhline(80, color=COND_COLORS["Healthy"], ls="--", lw=0.9, label="Healthy (80)")
a.axhline(60, color=COND_COLORS["Warning"], ls="--", lw=0.9, label="Warning (60)")
a.set(xlabel="Sample index", ylabel="Health score (0–100)", ylim=(40, 100))
a.legend(loc="lower right", ncol=2)

# (b) normalised sensor trends
z = (data[features] - data[features].mean()) / data[features].std()
for n, s in smooth(z).items():
    b.plot(s, color=SENSOR_COLORS[n], lw=1.1, label=n)
b.set(xlabel="Sample index", ylabel="Standardised value (z-score)", ylim=(-1, 1.5))
b.legend(ncol=3, loc="upper right", handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)

# (c) condition share per 50-sample window
share = pd.crosstab(data.index // 50, data["Condition"], normalize="index") * 100
for cond in LABELS[::-1]:
    c.plot(share.index * 50 + 25, share[cond], marker="o", ms=3, lw=1.2,
           color=COND_COLORS[cond], label=cond)
c.set(xlabel="Sample index", ylabel="Share of readings (%)", ylim=(-3, 105))
c.set_yticks(range(0, 101, 20))
c.legend(loc="upper right", ncol=3, borderaxespad=0.2)

# (d) individual sensor health scores
for n, s in smooth(scores).items():
    d.plot(s, color=SENSOR_COLORS[n], lw=1.1, label=n)
d.set(xlabel="Sample index", ylabel="Sensor health score (0–100)", ylim=(35, 100))
d.legend(ncol=3, loc="lower right", handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)

for ax, tag in zip(axes.flat, "abcd"):
    ax.set_title(f"({tag})", loc="left", fontweight="bold")

fig.savefig("equipment_health_figure.png")
fig.savefig("equipment_health_figure.pdf")  # vector, for LaTeX/journals
plt.show()
