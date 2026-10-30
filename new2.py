import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import time

# ================================================================ CONFIG
SENSORS = {
    "Temperature": (70, 10, 30, 120, 50, 75, 2, 4, 0.20),
    "Vibration":   (4, 1.5, 0, 10, 0, 0, 0, 10, 0.30),
    "Pressure":    (100, 8, 50, 150, 100, 100, 5, 5, 0.15),
    "RPM":         (1500, 120, 500, 2000, 1400, 1600, 0.5, 0.5, 0.15),
    "Current":     (30, 5, 5, 60, 20, 40, 4, 4, 0.20),
}

SENSOR_COLORS = {
    "Temperature": "#0072B2",
    "Vibration": "#D55E00",
    "Pressure": "#009E73",
    "RPM": "#CC79A7",
    "Current": "#E69F00"
}

COND_COLORS = {"Healthy": "#009E73", "Warning": "#E69F00", "Critical": "#D55E00"}

# ================================================================ FUNCTIONS
def band_score(x, lo, hi, k_low, k_high):
    """Calculate health score for a single value"""
    penalty = k_low * max(0, lo - x) + k_high * max(0, x - hi)
    return max(0, 100 - penalty)

def generate_sensor_value(name, seed):
    """Generate a random sensor value"""
    rng = np.random.default_rng(seed)
    m, s, lo, hi, *_ = SENSORS[name]
    val = rng.normal(m, s)
    return np.clip(val, lo, hi)

def calculate_health(temp, vibration, pressure, rpm, current):
    """Calculate overall health score"""
    temp_score = band_score(temp, 50, 75, 2, 4)
    vib_score = band_score(vibration, 0, 0, 0, 10)
    pres_score = band_score(pressure, 100, 100, 5, 5)
    rpm_score = band_score(rpm, 1400, 1600, 0.5, 0.5)
    curr_score = band_score(current, 20, 40, 4, 4)
    
    return (temp_score * 0.20 + vib_score * 0.30 + pres_score * 0.15 + 
            rpm_score * 0.15 + curr_score * 0.20)

def get_condition(health_score):
    """Determine condition status"""
    if health_score >= 80:
        return "Healthy"
    elif health_score >= 60:
        return "Warning"
    else:
        return "Critical"

# ================================================================ GENERATE INITIAL DATA
print("Generating 1000 samples of sensor data...")
N = 1000
data = {
    "Temperature": [],
    "Vibration": [],
    "Pressure": [],
    "RPM": [],
    "Current": [],
    "Health_Score": [],
    "Condition": []
}

rng = np.random.default_rng(42)
for i in range(N):
    temp = np.clip(rng.normal(70, 10), 30, 120)
    vib = np.clip(rng.normal(4, 1.5), 0, 10)
    pres = np.clip(rng.normal(100, 8), 50, 150)
    rpm = np.clip(rng.normal(1500, 120), 500, 2000)
    curr = np.clip(rng.normal(30, 5), 5, 60)
    health = calculate_health(temp, vib, pres, rpm, curr)
    cond = get_condition(health)
    
    data["Temperature"].append(temp)
    data["Vibration"].append(vib)
    data["Pressure"].append(pres)
    data["RPM"].append(rpm)
    data["Current"].append(curr)
    data["Health_Score"].append(health)
    data["Condition"].append(cond)

df = pd.DataFrame(data)

# ================================================================ ML MODEL
print("Training Random Forest model...")
features = ["Temperature", "Vibration", "Pressure", "RPM", "Current"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["Condition"], test_size=0.2, random_state=42, stratify=df["Condition"]
)
model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
pred = model.predict(X_test)

print(f"Model accuracy: {accuracy_score(y_test, pred):.2%}")
print(classification_report(y_test, pred, zero_division=0))

importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nSensor importance:\n", importance.round(3))

# ================================================================ CREATE ANIMATED FIGURE
print("Creating interactive animated dashboard...")

def smooth_series(series, window=25):
    """Apply moving average"""
    return pd.Series(series).rolling(window=window, center=True).mean()

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("(a) Health Score Over Time", "(b) Sensor Readings (Normalised)",
                    "(c) Condition Share per 50-sample Window", "(d) Individual Sensor Health"),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Calculate smoothed values
health_smooth = smooth_series(df["Health_Score"])
sensor_means = {s: df[s].mean() for s in features}
sensor_stds = {s: df[s].std() for s in features}

# Normalise sensors
normalized_sensors = {}
for s in features:
    normalized_sensors[s] = (df[s] - sensor_means[s]) / sensor_stds[s]

# Calculate condition share per 50-sample window
window_size = 50
condition_share = {"Healthy": [], "Warning": [], "Critical": []}
window_indices = []

for i in range(0, len(df), window_size):
    window_data = df.iloc[i:i+window_size]["Condition"]
    for cond in ["Healthy", "Warning", "Critical"]:
        count = (window_data == cond).sum()
        condition_share[cond].append(count / len(window_data) * 100)
    window_indices.append(i + window_size/2)

# Calculate individual sensor health scores
sensor_health = {}
for s in features:
    m, std, lo, hi, ideal_lo, ideal_hi, k_low, k_high, weight = SENSORS[s]
    sensor_health[s] = [band_score(val, ideal_lo, ideal_hi, k_low, k_high) for val in df[s]]
    sensor_health[s] = smooth_series(sensor_health[s])

# ================================================================ ANIMATION FRAMES
n_frames = len(df)
frames = []

for frame_idx in range(50, n_frames, 10):  # Skip first 50, update every 10 samples
    frame_data = []
    
    # (a) Health Score
    indices = list(range(frame_idx))
    frame_data.append(go.Scatter(
        x=indices, y=df["Health_Score"][:frame_idx],
        mode='lines', name='Health Score', line=dict(color="#0072B2", width=2),
        fill='tozeroy', fillcolor='rgba(0, 114, 178, 0.1)'
    ))
    frame_data.append(go.Scatter(
        x=indices, y=[80]*len(indices),
        mode='lines', name='Healthy (80)', line=dict(color="#009E73", width=2, dash='dash')
    ))
    frame_data.append(go.Scatter(
        x=indices, y=[60]*len(indices),
        mode='lines', name='Warning (60)', line=dict(color="#E69F00", width=2, dash='dash')
    ))
    
    # (b) Normalised Sensors
    for s in features:
        normalized_data = (df[s][:frame_idx] - sensor_means[s]) / sensor_stds[s]
        frame_data.append(go.Scatter(
            x=list(range(frame_idx)), y=normalized_data,
            mode='lines', name=s, line=dict(color=SENSOR_COLORS[s], width=1.5)
        ))
    
    # (c) Condition Share
    n_windows = max(1, frame_idx // window_size)
    for cond, color in [("Healthy", "#009E73"), ("Warning", "#E69F00"), ("Critical", "#D55E00")]:
        frame_data.append(go.Scatter(
            x=window_indices[:n_windows], y=condition_share[cond][:n_windows],
            mode='lines+markers', name=cond, line=dict(color=color, width=2),
            marker=dict(size=4)
        ))
    
    # (d) Sensor Health Scores
    for s in features:
        frame_data.append(go.Scatter(
            x=list(range(frame_idx)), y=sensor_health[s][:frame_idx],
            mode='lines', name=s, line=dict(color=SENSOR_COLORS[s], width=1.5)
        ))
    
    frames.append(go.Frame(data=frame_data, name=str(frame_idx)))

# Add initial data
fig.add_trace(go.Scatter(x=[0], y=[df["Health_Score"].iloc[0]], name='Health Score',
                         line=dict(color="#0072B2", width=2), fill='tozeroy', fillcolor='rgba(0, 114, 178, 0.1)'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=[0], y=[80], name='Healthy (80)',
                         line=dict(color="#009E73", width=2, dash='dash')),
              row=1, col=1)
fig.add_trace(go.Scatter(x=[0], y=[60], name='Warning (60)',
                         line=dict(color="#E69F00", width=2, dash='dash')),
              row=1, col=1)

for s in features:
    fig.add_trace(go.Scatter(x=[0], y=[0], name=s, line=dict(color=SENSOR_COLORS[s], width=1.5)),
                  row=1, col=2)

for cond, color in [("Healthy", "#009E73"), ("Warning", "#E69F00"), ("Critical", "#D55E00")]:
    fig.add_trace(go.Scatter(x=[0], y=[0], name=cond, line=dict(color=color, width=2),
                             marker=dict(size=4)),
                  row=2, col=1)

for s in features:
    fig.add_trace(go.Scatter(x=[0], y=[0], name=s, line=dict(color=SENSOR_COLORS[s], width=1.5)),
                  row=2, col=2)

# Update layout
fig.update_layout(
    title_text="<b>Equipment Health Monitoring Dashboard</b>",
    showlegend=True,
    height=900,
    width=1400,
    hovermode='x unified',
    template='plotly_white',
    updatemenus=[
        dict(type='buttons', showactive=False, x=0.05, y=1.08,
             buttons=[
                 dict(label='▶ Play', method='animate',
                      args=[None, {'frame': {'duration': 100, 'redraw': True},
                                   'fromcurrent': True, 'transition': {'duration': 50}}]),
                 dict(label='⏸ Pause', method='animate',
                      args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                     'mode': 'immediate', 'transition': {'duration': 0}}]),
                 dict(label='↻ Reset', method='animate',
                      args=[[frames[0].name], {'frame': {'duration': 0, 'redraw': True},
                                               'mode': 'immediate', 'transition': {'duration': 0}}])
             ])
    ]
)

# Update axes
fig.update_xaxes(title_text="Sample index", row=1, col=1)
fig.update_yaxes(title_text="Health score (0–100)", row=1, col=1, range=[40, 100])

fig.update_xaxes(title_text="Sample index", row=1, col=2)
fig.update_yaxes(title_text="Standardised value (z-score)", row=1, col=2, range=[-1.5, 1.5])

fig.update_xaxes(title_text="Sample index", row=2, col=1)
fig.update_yaxes(title_text="Share of readings (%)", row=2, col=1, range=[-5, 105])

fig.update_xaxes(title_text="Sample index", row=2, col=2)
fig.update_yaxes(title_text="Sensor health score (0–100)", row=2, col=2, range=[30, 100])

# Add frames
fig.frames = frames

print("Opening dashboard in browser...")
fig.show()