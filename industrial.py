"""
AI-Based Industrial Equipment Health Monitor
--------------------------------------------
Run:   streamlit run equipment_dashboard.py
Needs: pip install streamlit streamlit-autorefresh pandas numpy plotly paho-mqtt

* A background simulator generates realistic multi-sensor data (load cycles,
  slow wear, random faults) so the graphs move up and down live.
* Health score = weighted penalty model + sudden-change (z-score) anomaly
  detector, smoothed with an EWMA. Remaining useful life (RUL) comes from the
  health trend.
* Set USE_MQTT=1 to ingest real data instead/as well (JSON payload with
  equipment_id, timestamp and every sensor key listed in SENSORS).
"""
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow the regular VS Code "Run Python File" button to launch this app too.
if not st.runtime.exists():
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                os.path.abspath(__file__),
                "--server.headless",
                "false",
                "--server.port",
                "8502",
            ]
        )
    )

from streamlit_autorefresh import st_autorefresh

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DB_PATH = "equipment_health.db"
TABLE = "readings"
HISTORY_BACKFILL = 150        # readings generated on first start so charts aren't empty
MAX_ROWS_PER_EQ = 600         # rows loaded per machine for the dashboard
FAULT_PROB = 0.003            # chance per reading that a machine starts a fault
WEAR_RATE = 0.0003            # slow degradation per reading
TS_FMT = "%Y-%m-%d %H:%M:%S"

USE_MQTT = os.getenv("USE_MQTT", "0") == "1"
BROKER_HOST = os.getenv("BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("BROKER_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "test/topic")

# mode: "high" = bad when above nominal, "low" = bad when below, "band" = bad both ways
# span: deviation from nominal that means full penalty ("trip")
# warn: deviation from nominal that raises a warning alert
# weight: share of the health score this sensor controls
# jitter/load/wear: simulator noise, load sensitivity, wear sensitivity
SENSORS = {
    "temperature":  dict(label="Motor Temp",    unit="°C",   icon="🌡", nominal=65,   span=30,   warn=15,   mode="high", weight=0.15, jitter=0.8,   load=1.5,  wear=0.25),
    "bearing_temp": dict(label="Bearing Temp",  unit="°C",   icon="🔥", nominal=55,   span=35,   warn=22,   mode="high", weight=0.14, jitter=0.6,   load=1.0,  wear=0.35),
    "vibration":    dict(label="Vibration",     unit="mm/s", icon="📳", nominal=2.0,  span=5.0,  warn=2.5,  mode="high", weight=0.20, jitter=0.12,  load=0.3,  wear=0.30),
    "current":      dict(label="Current",       unit="A",    icon="⚡", nominal=15,   span=20,   warn=12,   mode="high", weight=0.12, jitter=0.3,   load=1.5,  wear=0.15),
    "voltage":      dict(label="Voltage",       unit="V",    icon="🔌", nominal=415,  span=50,   warn=30,   mode="band", weight=0.06, jitter=1.5,   load=-0.2, wear=0.0),
    "power_factor": dict(label="Power Factor",  unit="",     icon="📐", nominal=0.90, span=0.20, warn=0.08, mode="low",  weight=0.05, jitter=0.005, load=-0.3, wear=-0.30),
    "rpm":          dict(label="RPM",           unit="rpm",  icon="🔄", nominal=1450, span=200,  warn=100,  mode="band", weight=0.08, jitter=4.0,   load=-0.5, wear=-0.10),
    "pressure":     dict(label="Pressure",      unit="bar",  icon="🧭", nominal=3.5,  span=2.0,  warn=1.0,  mode="band", weight=0.08, jitter=0.05,  load=0.2,  wear=-0.15),
    "flow":         dict(label="Flow",          unit="m³/h", icon="💧", nominal=120,  span=60,   warn=30,   mode="band", weight=0.06, jitter=1.5,   load=0.2,  wear=-0.20),
    "noise_db":     dict(label="Acoustic Noise", unit="dB",  icon="🔊", nominal=70,   span=25,   warn=12,   mode="high", weight=0.09, jitter=0.8,   load=0.5,  wear=0.30),
}

# Per-machine nominal values (anything missing falls back to SENSORS[...]["nominal"])
PROFILES = {
    "MOTOR-01":      {"temperature": 65, "current": 15, "rpm": 1450},
    "PUMP-01":       {"pressure": 4.2, "flow": 140, "rpm": 1480, "current": 18},
    "COMPRESSOR-01": {"pressure": 7.0, "flow": 90, "rpm": 1620, "current": 22, "temperature": 72},
    "FAN-01":        {"rpm": 1200, "flow": 200, "pressure": 1.5, "current": 9, "temperature": 55},
    "GEARBOX-01":    {"rpm": 900, "current": 12, "bearing_temp": 60, "vibration": 2.4, "pressure": 2.5, "flow": 60},
}

# Fault signatures: additive effect at peak severity
FAULTS = {
    "bearing_wear": {"vibration": 4.0, "bearing_temp": 22, "noise_db": 14, "temperature": 6},
    "overload":     {"current": 14, "temperature": 20, "rpm": -90, "power_factor": -0.08},
    "cavitation":   {"pressure": -1.4, "flow": -45, "vibration": 2.5, "noise_db": 12},
    "overheating":  {"temperature": 28, "bearing_temp": 15, "current": 4},
    "voltage_sag":  {"voltage": -45, "current": 6, "rpm": -50},
}

OVERVIEW = "🏠 Overview (all equipment)"
STATUS_ICON = {"NORMAL": "🟢", "GOOD": "🟢", "WARNING": "🟡", "CRITICAL": "🟠", "SEVERE": "🔴"}
STATUS_COLOR = {"NORMAL": "#2ecc71", "GOOD": "#7bd88f", "WARNING": "#f1c40f",
                "CRITICAL": "#e67e22", "SEVERE": "#e74c3c"}
GRID = "rgba(128,128,128,0.2)"


def nominal_of(eq, key):
    return PROFILES.get(eq, {}).get(key, SENSORS[key]["nominal"])


def status_from_health(score):
    if score >= 90:
        return "NORMAL"
    if score >= 70:
        return "GOOD"
    if score >= 50:
        return "WARNING"
    if score >= 30:
        return "CRITICAL"
    return "SEVERE"


# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------
def _connect():
    return sqlite3.connect(DB_PATH, timeout=10)


def init_db():
    cols = ", ".join(f"{k} REAL NOT NULL" for k in SENSORS)
    conn = _connect()
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE} "
            f"(equipment_id TEXT NOT NULL, timestamp TEXT NOT NULL, {cols})"
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_eq_ts ON {TABLE}(equipment_id, timestamp)")
        conn.commit()
    finally:
        conn.close()


def insert_rows(rows):
    names = ["equipment_id", "timestamp"] + list(SENSORS)
    sql = f"INSERT INTO {TABLE} ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})"
    conn = _connect()
    try:
        conn.executemany(sql, [[r[n] for n in names] for r in rows])
        conn.commit()
    finally:
        conn.close()


def prune_db(keep=20000):
    conn = _connect()
    try:
        conn.execute(
            f"DELETE FROM {TABLE} WHERE rowid NOT IN "
            f"(SELECT rowid FROM {TABLE} ORDER BY rowid DESC LIMIT ?)", (keep,))
        conn.commit()
    finally:
        conn.close()


def count_rows():
    conn = _connect()
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    finally:
        conn.close()


def load_data(limit=MAX_ROWS_PER_EQ):
    conn = _connect()
    try:
        df = pd.read_sql(
            f"""SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY equipment_id
                                                 ORDER BY timestamp DESC) AS rn
                    FROM {TABLE}) WHERE rn <= ? ORDER BY equipment_id, timestamp""",
            conn, params=(limit,))
    finally:
        conn.close()
    df["time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["time"]).drop(columns=["rn"])
    return df.drop_duplicates(["equipment_id", "time"], keep="last").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Simulator (background thread -> writes into the same DB as MQTT would)
# ----------------------------------------------------------------------------
def generate_reading(eq, state):
    t = state["t"]
    state["t"] += 1
    load = 1 + 0.08 * math.sin(t / 25 + state["phase"]) + 0.03 * math.sin(t / 6.5)

    sev, effects, fault = 0.0, {}, state["fault"]
    if fault:
        progress = fault["elapsed"] / fault["duration"]
        sev = math.sin(math.pi * min(progress, 1.0))     # ramps up, peaks, recovers
        effects = FAULTS[fault["type"]]
        fault["elapsed"] += 1
        if fault["elapsed"] >= fault["duration"]:
            state["fault"] = None

    row = {"equipment_id": eq}
    for key, s in SENSORS.items():
        value = (nominal_of(eq, key)
                 + (load - 1) * s["load"] * s["span"]
                 + state["wear"] * s["wear"] * s["span"]
                 + sev * effects.get(key, 0.0)
                 + random.gauss(0, s["jitter"]))
        value = max(value, 0.0)
        if key == "power_factor":
            value = min(value, 1.0)
        row[key] = round(value, 3)
    state["wear"] = min(1.0, state["wear"] + WEAR_RATE)
    return row


class Simulator:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.interval = 2.0
        self.ticks = 0
        self.state = {
            eq: {"wear": random.uniform(0.0, 0.15), "phase": random.uniform(0, 6.28),
                 "t": 0, "fault": None}
            for eq in PROFILES
        }
        if count_rows() < HISTORY_BACKFILL:                  # pre-fill so charts start populated
            now = datetime.now()
            for i in range(HISTORY_BACKFILL):
                self.step(now - timedelta(seconds=(HISTORY_BACKFILL - i) * self.interval))
        threading.Thread(target=self._loop, daemon=True).start()

    def step(self, ts):
        rows = []
        with self.lock:
            self.ticks += 1
            for eq, s in self.state.items():
                if s["fault"] is None and random.random() < FAULT_PROB:
                    s["fault"] = {"type": random.choice(list(FAULTS)), "elapsed": 0,
                                  "duration": random.randint(40, 90)}
                row = generate_reading(eq, s)
                row["timestamp"] = ts.strftime(TS_FMT)
                rows.append(row)
        insert_rows(rows)
        if self.ticks % 300 == 0:
            prune_db()

    def _loop(self):
        while True:
            time.sleep(self.interval)
            if self.running:
                try:
                    self.step(datetime.now())
                except Exception as exc:      # never let the thread die
                    print(f"[!] simulator error: {exc}")

    def inject(self, eq, fault_type, duration=60):
        with self.lock:
            self.state[eq]["fault"] = {"type": fault_type, "elapsed": 0, "duration": duration}

    def maintain(self, eq):
        with self.lock:
            self.state[eq]["wear"] = 0.0
            self.state[eq]["fault"] = None


@st.cache_resource
def get_simulator():
    init_db()
    return Simulator()


# ----------------------------------------------------------------------------
# Optional MQTT ingestion (real sensors)
# ----------------------------------------------------------------------------
@st.cache_resource
def start_mqtt():
    import paho.mqtt.client as mqtt

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            client.subscribe(MQTT_TOPIC)
            print(f"[+] MQTT connected, subscribed to '{MQTT_TOPIC}'")

    def on_message(client, userdata, message):
        try:
            p = json.loads(message.payload.decode("utf-8"))
            if {"equipment_id", "timestamp", *SENSORS}.issubset(p):
                insert_rows([{"equipment_id": p["equipment_id"], "timestamp": p["timestamp"],
                              **{k: float(p[k]) for k in SENSORS}}])
        except Exception as exc:
            print(f"[!] bad MQTT payload: {exc}")

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect, client.on_message = on_connect, on_message
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()
    return client


# ----------------------------------------------------------------------------
# AI health scoring
# ----------------------------------------------------------------------------
def score_health(df):
    """Adds pen_<sensor> (health points lost), anomaly, health_raw, health."""
    df = df.copy()
    total_penalty = pd.Series(0.0, index=df.index)

    # 1) Weighted deviation-from-nominal penalty model
    for key, s in SENSORS.items():
        nominal = df["equipment_id"].map(lambda e: nominal_of(e, key)).astype(float)
        dev = df[key] - nominal
        if s["mode"] == "high":
            dev = dev.clip(lower=0)
        elif s["mode"] == "low":
            dev = (-dev).clip(lower=0)
        else:
            dev = dev.abs()
        p = (dev / s["span"]).clip(0, 1) ** 1.5          # gentle near nominal, steep near trip
        df[f"pen_{key}"] = 100 * s["weight"] * p
        total_penalty += s["weight"] * p

    # 2) Sudden-change detector: rolling z-score against the machine's own recent past
    flags = pd.DataFrame(index=df.index)
    for key in SENSORS:
        g = df.groupby("equipment_id")[key]
        mean = g.transform(lambda s: s.shift(1).rolling(60, min_periods=20).mean())
        std = g.transform(lambda s: s.shift(1).rolling(60, min_periods=20).std())
        z = (df[key] - mean) / std.replace(0, np.nan)
        flags[key] = (z.abs() > 3).astype(float)
    df["anomaly"] = flags.mean(axis=1)                    # share of sensors behaving oddly

    # 3) Combine + smooth
    df["health_raw"] = (100 * (1 - total_penalty.clip(0, 1)) - 15 * df["anomaly"]).clip(0, 100)
    df["health"] = df.groupby("equipment_id")["health_raw"].transform(
        lambda s: s.ewm(span=4, adjust=False).mean())
    return df


def estimate_rul(d, floor=50, window=60):
    """Seconds until health trend reaches `floor`; None if not degrading."""
    tail = d.tail(window)
    if len(tail) < 15:
        return None
    x = (tail["time"] - tail["time"].iloc[0]).dt.total_seconds().to_numpy()
    slope, _ = np.polyfit(x, tail["health"].to_numpy(), 1)
    current = tail["health"].iloc[-1]
    if current <= floor:
        return 0.0
    if slope > -1e-3:
        return None
    return (current - floor) / -slope


def fmt_duration(seconds):
    if seconds is None:
        return "Stable"
    if seconds <= 0:
        return "Now"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def top_contributor(row):
    pens = {k: row[f"pen_{k}"] for k in SENSORS}
    key = max(pens, key=pens.get)
    return key, pens[key]


def limit_lines(eq, key):
    s, n = SENSORS[key], nominal_of(eq, key)
    if s["mode"] == "high":
        return [n + s["warn"]]
    if s["mode"] == "low":
        return [n - s["warn"]]
    return [n - s["warn"], n + s["warn"]]


def sensor_alerts(eq, row):
    msgs = []
    for key, s in SENSORS.items():
        v = row[key]
        high, low = None, None
        lines = limit_lines(eq, key)
        if s["mode"] in ("high", "band") and v > max(lines):
            high = max(lines)
        if s["mode"] in ("low", "band") and v < min(lines):
            low = min(lines)
        if high is not None:
            msgs.append(f"{s['label']} {v:.2f} {s['unit']} is above limit {high:g}")
        if low is not None:
            msgs.append(f"{s['label']} {v:.2f} {s['unit']} is below limit {low:g}")
    return msgs


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def health_chart(d, color, avg, height=300):
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=50, fillcolor="#e74c3c", opacity=0.08, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor="#f1c40f", opacity=0.10, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor="#2ecc71", opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(x=d["time"], y=d["health"], mode="lines", name="Health",
                             line=dict(color=color, width=3),
                             hovertemplate="%{x|%H:%M:%S}<br>Health: %{y:.1f}<extra></extra>"))
    if avg > 1 and len(d) >= avg:
        fig.add_trace(go.Scatter(x=d["time"], y=d["health"].rolling(avg).mean(), mode="lines",
                                 name=f"{avg}-pt average",
                                 line=dict(color="#7f7f7f", width=2, dash="dash")))
    last = d.iloc[-1]
    fig.add_trace(go.Scatter(x=[last["time"]], y=[last["health"]], mode="markers",
                             marker=dict(size=11, color=color, line=dict(width=2, color="white")),
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=dict(text="Equipment Health Over Time", x=0.01), height=height,
                      margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                      xaxis=dict(type="date", tickformat="%H:%M:%S", gridcolor=GRID),
                      yaxis=dict(title="Health (0-100)", range=[0, 100], gridcolor=GRID),
                      uirevision="health")
    return fig


def sensor_chart(d, eq, key, avg, show_limits):
    s = SENSORS[key]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["time"], y=d[key], mode="lines", name=s["label"],
        line=dict(width=2.2, color="#1f77b4"),
        hovertemplate=f"%{{x|%H:%M:%S}}<br>{s['label']}: %{{y:.2f}} {s['unit']}<extra></extra>"))
    if avg > 1 and len(d) >= avg:
        fig.add_trace(go.Scatter(x=d["time"], y=d[key].rolling(avg).mean(), mode="lines",
                                 name=f"{avg}-pt average",
                                 line=dict(color="#7f7f7f", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[d["time"].iloc[-1]], y=[d[key].iloc[-1]], mode="markers",
                             marker=dict(size=10, color="#1f77b4", line=dict(width=2, color="white")),
                             showlegend=False, hoverinfo="skip"))
    values = d[key].tolist()
    if show_limits:
        for lim in limit_lines(eq, key):
            values.append(lim)
            fig.add_hline(y=lim, line_dash="dash", line_color="red",
                          annotation_text=f"Limit {lim:g}", annotation_position="top left")
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.1 or 1.0
    fig.update_layout(title=dict(text=f"{s['icon']} {s['label']}", x=0.01), height=280,
                      margin=dict(l=10, r=10, t=45, b=10), hovermode="x unified",
                      showlegend=avg > 1,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                      xaxis=dict(type="date", tickformat="%H:%M:%S", gridcolor=GRID),
                      yaxis=dict(title=s["unit"], range=[lo - pad, hi + pad], gridcolor=GRID),
                      uirevision=f"{eq}_{key}")
    return fig


# ----------------------------------------------------------------------------
# Page setup + sidebar
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Equipment Health Monitor", page_icon="🏭", layout="wide")
init_db()
sim = get_simulator()
if USE_MQTT:
    try:
        start_mqtt()
    except Exception:
        st.sidebar.error("MQTT broker not reachable. Showing stored/simulated data only.")


def open_equipment(eq_id):
    st.session_state["nav"] = eq_id


def go_overview():
    st.session_state["nav"] = OVERVIEW


st.sidebar.header("⚙️ Controls")
if st.sidebar.toggle("Live auto-refresh", value=True):
    refresh_s = st.sidebar.slider("Refresh every (seconds)", 1, 10, 2)
    st_autorefresh(interval=refresh_s * 1000, key="refresh")

with st.sidebar.expander("🧪 Simulator", expanded=True):
    sim.running = st.toggle("Simulator running", value=sim.running)
    sim.interval = st.slider("Seconds per reading", 0.5, 5.0, float(sim.interval), 0.5)
    target = st.selectbox("Machine", list(PROFILES))
    fault_type = st.selectbox("Fault type", list(FAULTS))
    b1, b2 = st.columns(2)
    if b1.button("💥 Inject fault", width='stretch'):
        sim.inject(target, fault_type)
    if b2.button("🛠 Maintenance", width='stretch'):
        sim.maintain(target)
    st.caption("Faults ramp up over ~1-3 min then recover. Maintenance resets wear.")

raw = load_data()
if raw.empty:
    st.title("🏭 Industrial Equipment Health Monitor")
    st.warning("No data yet. Waiting for readings...")
    st.stop()

all_df = score_health(raw)
equipment_ids = sorted(all_df["equipment_id"].unique())
nav_options = [OVERVIEW] + equipment_ids
if st.session_state.get("nav") not in nav_options:
    st.session_state["nav"] = OVERVIEW
st.sidebar.selectbox("📍 Go to", nav_options, key="nav")
st.sidebar.caption(f"Last update: {time.strftime('%H:%M:%S')}")

latest_by_eq = all_df.groupby("equipment_id").tail(1).set_index("equipment_id")
latest_by_eq["status"] = latest_by_eq["health"].apply(status_from_health)

st.title("🏭 Industrial Equipment Health Monitor")


# ----------------------------------------------------------------------------
# Overview page
# ----------------------------------------------------------------------------
def render_overview():
    st.caption("All equipment at a glance. Click **Open details** to drill down.")
    statuses = latest_by_eq["status"]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Equipment", len(latest_by_eq))
    s2.metric("Healthy", int(statuses.isin(["NORMAL", "GOOD"]).sum()))
    s3.metric("Warning", int((statuses == "WARNING").sum()))
    s4.metric("Critical / Severe", int(statuses.isin(["CRITICAL", "SEVERE"]).sum()))
    st.divider()

    st.subheader("🔧 Equipment")
    per_row = 3
    for start in range(0, len(equipment_ids), per_row):
        cols = st.columns(per_row)
        for col, eq in zip(cols, equipment_ids[start:start + per_row]):
            row, status = latest_by_eq.loc[eq], latest_by_eq.loc[eq, "status"]
            hist = all_df[all_df["equipment_id"] == eq].tail(80)
            prev = hist["health"].iloc[-2] if len(hist) > 1 else row["health"]
            with col, st.container(border=True):
                st.markdown(f"### {STATUS_ICON[status]} {eq}")
                st.caption(f"Status: **{status}** • Updated: {row['timestamp']}")
                m1, m2 = st.columns(2)
                m1.metric("Health", f"{row['health']:.0f}/100", f"{row['health'] - prev:+.1f}")
                m2.metric("Temp", f"{row['temperature']:.1f} °C")
                m3, m4 = st.columns(2)
                m3.metric("Vibration", f"{row['vibration']:.2f} mm/s")
                m4.metric("Current", f"{row['current']:.1f} A")
                spark = go.Figure(go.Scatter(
                    x=hist["time"], y=hist["health"], mode="lines", fill="tozeroy",
                    line=dict(color=STATUS_COLOR[status], width=2.5),
                    hovertemplate="%{x|%H:%M:%S}<br>Health: %{y:.1f}<extra></extra>"))
                spark.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0),
                                    xaxis=dict(visible=False, type="date"),
                                    yaxis=dict(visible=False, range=[0, 100]))
                st.plotly_chart(spark, width='stretch', key=f"spark_{eq}",
                                config={"displayModeBar": False})
                st.button("🔍 Open details", key=f"open_{eq}", on_click=open_equipment,
                          args=(eq,), width='stretch')

    st.divider()
    st.subheader("📊 Fleet Comparison")
    left, right = st.columns(2)
    with left:
        bar = px.bar(latest_by_eq.reset_index(), x="equipment_id", y="health", color="status",
                     color_discrete_map=STATUS_COLOR, text="health", title="Current Health Score")
        bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        bar.update_yaxes(range=[0, 110], title="Health (0-100)", gridcolor=GRID)
        bar.update_xaxes(title="Equipment")
        st.plotly_chart(bar, width='stretch', key="fleet_bar")
    with right:
        trend = px.line(all_df.groupby("equipment_id").tail(150), x="time", y="health",
                        color="equipment_id", title="Health Trend (all equipment)")
        trend.update_yaxes(range=[0, 100], title="Health (0-100)", gridcolor=GRID)
        trend.update_xaxes(title="Time", tickformat="%H:%M:%S", gridcolor=GRID)
        trend.update_layout(hovermode="x unified", uirevision="fleet")
        st.plotly_chart(trend, width='stretch', key="fleet_trend")

    st.subheader("📋 Latest Readings")
    rows = []
    for eq in equipment_ids:
        r = latest_by_eq.loc[eq]
        key, pts = top_contributor(r)
        rows.append({"Equipment": eq, "Status": r["status"], "Health": r["health"],
                     "Main contributor": f"{SENSORS[key]['label']} (-{pts:.1f})" if pts > 0.5 else "-",
                     "RUL": fmt_duration(estimate_rul(all_df[all_df["equipment_id"] == eq])),
                     **{SENSORS[k]["label"]: r[k] for k in SENSORS}, "Time": r["timestamp"]})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True,
                 column_config={"Health": st.column_config.ProgressColumn(
                     "Health", min_value=0, max_value=100, format="%.0f")})

    st.subheader("⚠️ Alerts")
    any_alert = False
    for eq in equipment_ids:
        r = latest_by_eq.loc[eq]
        key, pts = top_contributor(r)
        why = f" Main contributor: **{SENSORS[key]['label']}** (-{pts:.1f} pts)." if pts > 0.5 else ""
        if r["health"] < 50:
            st.error(f"**{eq}**: critical condition (health {r['health']:.0f}).{why}")
            any_alert = True
        elif r["health"] < 70:
            st.warning(f"**{eq}**: abnormal behaviour (health {r['health']:.0f}).{why}")
            any_alert = True
    if not any_alert:
        st.success("All equipment is operating within the configured health range.")


# ----------------------------------------------------------------------------
# Detail page
# ----------------------------------------------------------------------------
def render_detail(eq):
    st.button("⬅ Back to overview", on_click=go_overview)
    df = all_df[all_df["equipment_id"] == eq].reset_index(drop=True)
    status = status_from_health(df.iloc[-1]["health"])
    st.header(f"{STATUS_ICON[status]} {eq} — {status}")
    fault = sim.state.get(eq, {}).get("fault")
    if fault:
        st.caption(f"🧪 Simulator: **{fault['type']}** in progress "
                   f"({fault['elapsed']}/{fault['duration']} readings)")

    c1, c2 = st.columns([1, 2])
    window = c1.select_slider("Show last N readings", [30, 60, 120, 300, 600], value=120,
                              key=f"win_{eq}")
    chosen = c2.multiselect("Sensors to chart", list(SENSORS),
                            default=["temperature", "vibration", "current", "bearing_temp"],
                            format_func=lambda k: f"{SENSORS[k]['icon']} {SENSORS[k]['label']}",
                            key=f"sens_{eq}")
    o1, o2 = st.columns(2)
    show_limits = o1.checkbox("Show warning limits", True, key=f"lim_{eq}")
    avg = o2.select_slider("Moving average (readings)", [0, 3, 5, 10, 20], value=0,
                           format_func=lambda v: "Off" if v == 0 else str(v), key=f"avg_{eq}")

    view = df.tail(int(window))
    latest = view.iloc[-1]
    prev = view.iloc[-2] if len(view) > 1 else latest

    top_key, top_pts = top_contributor(latest)
    rul = estimate_rul(df)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Health", f"{latest['health']:.0f}/100", f"{latest['health'] - prev['health']:+.1f}")
    k2.metric("Temperature", f"{latest['temperature']:.1f} °C",
              f"{latest['temperature'] - prev['temperature']:+.1f}", delta_color="inverse")
    k3.metric("Vibration", f"{latest['vibration']:.2f} mm/s",
              f"{latest['vibration'] - prev['vibration']:+.2f}", delta_color="inverse")
    k4.metric("Est. time to health 50", fmt_duration(rul))
    k5.metric("Anomalous sensors", f"{latest['anomaly'] * len(SENSORS):.0f} / {len(SENSORS)}")

    g_col, h_col = st.columns([1, 2])
    with g_col:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=float(latest["health"]), title={"text": "Health Score"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": STATUS_COLOR[status]},
                   "steps": [{"range": [0, 30], "color": "#f5b7b1"},
                             {"range": [30, 50], "color": "#fad7a0"},
                             {"range": [50, 70], "color": "#f9e79f"},
                             {"range": [70, 100], "color": "#abebc6"}]}))
        gauge.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
        st.plotly_chart(gauge, width='stretch', key=f"gauge_{eq}")
    with h_col:
        st.plotly_chart(health_chart(view, STATUS_COLOR[status], avg),
                        width='stretch', key=f"health_{eq}")

    st.subheader("🧠 Why this score?")
    contrib = pd.DataFrame({"sensor": [SENSORS[k]["label"] for k in SENSORS],
                            "points_lost": [latest[f"pen_{k}"] for k in SENSORS]}
                           ).sort_values("points_lost")
    cb = px.bar(contrib, x="points_lost", y="sensor", orientation="h",
                title="Health points lost per sensor (latest reading)")
    cb.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), uirevision=f"c_{eq}")
    cb.update_xaxes(gridcolor=GRID, title="Points lost")
    st.plotly_chart(cb, width='stretch', key=f"contrib_{eq}")

    st.subheader("📈 Sensor History")
    if not chosen:
        st.info("Select at least one sensor above.")
    for i in range(0, len(chosen), 2):
        cols = st.columns(2)
        for col, key in zip(cols, chosen[i:i + 2]):
            with col:
                st.plotly_chart(sensor_chart(view, eq, key, avg, show_limits),
                                width='stretch', key=f"{key}_{eq}")

    st.subheader("⚠️ AI Alerts")
    if latest["health"] < 50:
        st.error("Critical equipment condition. Schedule maintenance immediately.")
    elif latest["health"] < 70:
        st.warning("Abnormal equipment behaviour detected.")
    else:
        st.success("Equipment operating within the configured health range.")
    if top_pts > 0.5:
        st.info(f"Main contributor: **{SENSORS[top_key]['label']}** (-{top_pts:.1f} pts)")
    for msg in sensor_alerts(eq, latest):
        st.warning(msg)

    with st.expander("📋 Raw data"):
        show = view.drop(columns=["time"] + [c for c in view.columns if c.startswith("pen_")])
        st.dataframe(show.iloc[::-1], width='stretch', hide_index=True)
        st.download_button("⬇ Download CSV", show.to_csv(index=False),
                           file_name=f"{eq}_data.csv", mime="text/csv")


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
if st.session_state["nav"] == OVERVIEW:
    render_overview()
else:
    render_detail(st.session_state["nav"])