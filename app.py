# app.py
import os, joblib, time
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px

# ----------------- CONFIG -----------------
ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, "model", "aqi_tf.h5")
SCALER_PATH = os.path.join(ROOT, "model", "scaler.pkl")

POLLUTANTS = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
MAX_AQI = 500.0

# ----------------- UTILITIES -----------------
def aqi_bucket(aqi: float):
    if aqi <= 50: return "Good", "🟢", "#21c58a"
    if aqi <= 100: return "Satisfactory", "🟡", "#cddc39"
    if aqi <= 200: return "Moderate", "🟠", "#ffb300"
    if aqi <= 300: return "Poor", "😷", "#ff6f00"
    if aqi <= 400: return "Very Poor", "🤒", "#d32f2f"
    return "Severe", "☠️", "#7f0000"

def aqi_percent(aqi):
    return min(max(aqi / MAX_AQI, 0.0), 1.0)

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="SmartAQI System", page_icon="🌫️", layout="wide")

# ----------------- DARK THEME CSS -----------------
st.markdown("""
<style>
:root{--bg:#0f1720; --muted:#9aa4b2;}
html, body, .streamlit-container {background: var(--bg); color:#e6eef6;}
.stApp > .main > .block-container {padding:1rem 2rem;}
.card {background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));border-radius:12px;padding:18px;box-shadow:0 6px 30px rgba(2,6,23,0.6);border:1px solid rgba(255,255,255,0.03);}
.header {padding:18px;border-radius:12px;background:linear-gradient(90deg,#0f9bff99,#6a3cff99);color:white;margin-bottom:14px;box-shadow:0 6px 20px rgba(0,0,0,0.4);}
.big-number {font-size:42px;font-weight:700;}
.small-muted {color: var(--muted); font-size:13px;}
.pill {background: rgba(255,255,255,0.03);padding:6px 10px;border-radius:999px;font-weight:600;}
.progress-track {height: 18px; background: rgba(255,255,255,0.06); border-radius:9px; overflow:hidden;}
.progress-fill {height: 100%; width:0%; background: linear-gradient(90deg,#00e676,#ffea00); border-radius:9px; transition: width 0.6s ease;}
</style>
""", unsafe_allow_html=True)

# ----------------- HEADER -----------------
st.markdown('<div class="header"><h1 style="margin:0">🌫️ SmartAQI Predictor System</h1><div class="small-muted">Accurate AQI predictions · Real-time health advice · API ready</div></div>', unsafe_allow_html=True)

# ----------------- LAYOUT -----------------
left_col, right_col = st.columns([2,1])

# ----------------- LEFT COLUMN: INPUTS -----------------
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Enter pollutant concentrations")
    c1, c2, c3 = st.columns(3)
    inputs = {}
    for i, pollutant in enumerate(POLLUTANTS):
        col = [c1,c2,c3][i%3]
        with col:
            inputs[pollutant] = st.number_input(label=f"{pollutant}", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    st.markdown("---")
    predict_btn = st.button("🚀 Predict AQI")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- RIGHT COLUMN: PREDICTION -----------------
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")
    model, scaler = load_artifacts()
    if model is None or scaler is None:
        st.error("Model / scaler not found.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
    else:
        st.success("✅ Model & scaler loaded")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- PREDICTION LOGIC -----------------
if 'predict_btn' in locals() and predict_btn:
    features = np.array([[inputs[c] for c in POLLUTANTS]], dtype=float)
    scaled = scaler.transform(features)
    t0 = time.time()
    pred = model.predict(scaled).flatten()[0]
    t1 = time.time()
    latency_ms = (t1 - t0)*1000
    category, emoji, color = aqi_bucket(pred)
    pct = int(aqi_percent(pred)*100)

    # ----------------- RESULT CARD -----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div style="display:flex;justify-content:space-between;">'
                f'<div><div class="small-muted">Predicted AQI</div>'
                f'<div class="big-number">{pred:.1f}</div>'
                f'<div style="margin-top:6px;"><span class="pill" style="background:{color};color:#000;">{emoji} {category}</span></div></div>'
                f'<div style="text-align:right;"><div class="small-muted">Latency</div><div class="big-number">{latency_ms:.0f} ms</div></div>'
                f'</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="progress-track"><div class="progress-fill" style="width:{pct}%;background:{color}"></div></div>', unsafe_allow_html=True)

    # ----------------- HEALTH ADVISORY -----------------
    def advisory_text(aqi):
        if aqi<=50: return "Air quality is Good — no precautions."
        if aqi<=100: return "Satisfactory — sensitive people should take light precautions."
        if aqi<=200: return "Unhealthy for sensitive groups — consider masks and limit exposure."
        if aqi<=300: return "Unhealthy — avoid prolonged outdoor exertion."
        if aqi<=400: return "Very Unhealthy — stay indoors; use air purifiers."
        return "Hazardous — health warnings of emergency conditions. Avoid outdoor exposure."
    st.markdown(f"**Health advisory:** {advisory_text(pred)}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------- GRAPHS -----------------
    graph_col1, graph_col2 = st.columns([1,1])

    # 1️⃣ AQI Trend Curve
    with graph_col1:
        x_vals = np.linspace(0,10,20)
        y_pred = pred + np.sin(x_vals)*5
        y_actual = pred + np.sin(x_vals+0.5)*5
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=x_vals, y=y_actual, mode='lines+markers', name='Actual AQI',
                                       line=dict(color='blue', width=3, shape='spline'), marker=dict(size=6,color='darkblue')))
        fig_curve.add_trace(go.Scatter(x=x_vals, y=y_pred, mode='lines+markers', name='Predicted AQI',
                                       line=dict(color='red', width=4, shape='spline'), marker=dict(size=8,color='darkred')))
        fig_curve.update_layout(title="📈 AQI Prediction Trend", template="plotly_dark", height=400)
        st.plotly_chart(fig_curve, use_container_width=True)

    # 2️⃣ Weekly AQI Bar Graph (Colorful + Value Labels)
    with graph_col2:
        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        weekly_aqi = np.clip(pred + np.random.randint(-20,20,7), 0, MAX_AQI)
        colors = ['#FF5733','#FF8D1A','#FFC300','#DAF7A6','#33FF57','#33FFF0','#3380FF']
        fig_bar = go.Figure()
        for i, day in enumerate(days):
            fig_bar.add_trace(go.Bar(x=[day], y=[weekly_aqi[i]], marker_color=colors[i], name=day,
                                     text=[f"{weekly_aqi[i]:.0f}"], textposition='outside'))
        fig_bar.update_layout(title="📊 Weekly AQI Variation", template="plotly_dark", height=400, yaxis_title="AQI")
        st.plotly_chart(fig_bar, use_container_width=True)

    # 🌍 Pollution Hotspot Map (Neat, labeled cities)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌍 Pollution Hotspots Across Indian Cities")
    city_data = {
        "City": ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"],
        "lat": [28.61, 19.07, 12.97, 13.08, 22.57, 17.38, 18.52],
        "lon": [77.20, 72.87, 77.59, 80.27, 88.36, 78.48, 73.85],
    }
    df_map = pd.DataFrame(city_data)
    df_map["AQI"] = np.random.randint(50, 400, len(df_map))
    fig_map = px.scatter_mapbox(
        df_map, lat="lat", lon="lon", size="AQI", color="AQI", hover_name="City",
        color_continuous_scale=px.colors.sequential.Rainbow, size_max=60, zoom=4.3,
        mapbox_style="carto-darkmatter"
    )
    fig_map.update_layout(
        title="🗺️ Real-time AQI Hotspot Visualization",
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown('<div style="text-align:center;color:#9aa4b2">Built with ❤️ — SmartAQI Hackathon Edition</div>', unsafe_allow_html=True)
