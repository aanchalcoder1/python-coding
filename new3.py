import sqlite3
import pandas as pd


def get_connection():
    return sqlite3.connect("local.db")


def initialize_database(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS equipment_data (
            equipment_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            temperature REAL NOT NULL,
            vibration REAL NOT NULL,
            current REAL NOT NULL,
            rpm REAL NOT NULL,
            pressure REAL NOT NULL,
            health_score REAL NOT NULL,
            status TEXT NOT NULL
        )
        """
    )

    row_count = connection.execute(
        "SELECT COUNT(*) FROM equipment_data"
    ).fetchone()[0]

    if row_count == 0:
        connection.executemany(
            """
            INSERT INTO equipment_data
            (equipment_id, timestamp, temperature, vibration, current,
             rpm, pressure, health_score, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("MOTOR-01", "2026-09-29 09:00", 62.5, 2.1, 12.4, 1450, 3.2, 92, "NORMAL"),
                ("MOTOR-01", "2026-09-29 10:00", 64.0, 2.4, 12.8, 1445, 3.1, 88, "GOOD"),
                ("PUMP-01", "2026-09-29 09:00", 71.2, 3.8, 15.1, 1380, 2.8, 68, "WARNING"),
                ("PUMP-01", "2026-09-29 10:00", 73.5, 4.2, 15.6, 1372, 2.7, 61, "WARNING"),
                ("COMPRESSOR-01", "2026-09-29 09:00", 58.4, 1.7, 10.8, 1620, 4.1, 95, "NORMAL"),
                ("COMPRESSOR-01", "2026-09-29 10:00", 59.1, 1.8, 10.9, 1618, 4.0, 94, "NORMAL"),
            ],
        )
        connection.commit()


def get_equipment_data(equipment_id):
    connection = get_connection()
    initialize_database(connection)

    query = """
        SELECT
            timestamp,
            temperature,
            vibration,
            current,
            rpm,
            pressure,
            health_score,
            status
        FROM equipment_data
        WHERE equipment_id = ?
        ORDER BY timestamp
    """

    df = pd.read_sql(query, connection, params=(equipment_id,))

    connection.close()

    return df
import streamlit as st
import plotly.express as px


st.set_page_config(
    page_title="Equipment Health Monitor",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Industrial Equipment Health Monitor")

# Sidebar
st.sidebar.header("Equipment")

equipment_id = st.sidebar.selectbox(
    "Select Equipment",
    ["MOTOR-01", "PUMP-01", "COMPRESSOR-01"]
)

# Get database data
df = get_equipment_data(equipment_id)

if df.empty:
    st.warning("No data available.")
    st.stop()

# Latest reading
latest = df.iloc[-1]

health = latest["health_score"]
temperature = latest["temperature"]
vibration = latest["vibration"]
current = latest["current"]

# Status
if health >= 90:
    status = "🟢 NORMAL"
elif health >= 70:
    status = "🟢 GOOD"
elif health >= 50:
    status = "🟡 WARNING"
elif health >= 30:
    status = "🟠 CRITICAL"
else:
    status = "🔴 SEVERE"

# KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Health Score",
    f"{health:.0f}/100"
)

col2.metric(
    "Temperature",
    f"{temperature:.1f} °C"
)

col3.metric(
    "Vibration",
    f"{vibration:.2f} mm/s"
)

col4.metric(
    "Current",
    f"{current:.2f} A"
)

st.subheader(f"Equipment Status: {status}")

# Health graph
st.subheader("📈 Health Score History")

fig_health = px.line(
    df,
    x="timestamp",
    y="health_score",
    markers=True,
    title="Equipment Health Over Time"
)

fig_health.update_yaxes(range=[0, 100])

st.plotly_chart(fig_health, use_container_width=True)

# Sensor graphs
col1, col2 = st.columns(2)

with col1:

    st.subheader("🌡 Temperature")

    fig_temp = px.line(
        df,
        x="timestamp",
        y="temperature",
        title="Temperature History"
    )

    st.plotly_chart(
        fig_temp,
        use_container_width=True
    )


with col2:

    st.subheader("📳 Vibration")

    fig_vibration = px.line(
        df,
        x="timestamp",
        y="vibration",
        title="Vibration History"
    )ppp'

    st.plotly_chart(
        fig_vibration,
        use_container_width=True
    )

# Current and RPM
col1, col2 = st.columns(2)

with col1:

    st.subheader("⚡ Motor Current")

    fig_current = px.line(
        df,
        x="timestamp",
        y="current",
        title="Motor Current"
    )

    st.plotly_chart(
        fig_current,
        use_container_width=True
    )


with col2:

    st.subheader("🔄 RPM")

    fig_rpm = px.line(
        df,
        x="timestamp",
        y="rpm",
        title="Motor RPM"
    )

    st.plotly_chart(
        fig_rpm,
        use_container_width=True
    )

# Alerts
st.subheader("⚠️ AI Alerts")

if health < 50:
    st.error(
        "Critical equipment condition. "
        "Maintenance should be investigated."
    )

elif health < 70:
    st.warning(
        "Abnormal equipment behavior detected."
    )

else:
    st.success(
        "Equipment operating within the configured health range."
    )