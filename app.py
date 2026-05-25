import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from datetime import datetime, timedelta

# ── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción de consumo eléctrico",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Predicción de consumo eléctrico en España")
st.markdown("Selecciona una fecha y hora para predecir el consumo eléctrico usando una LSTM entrenada sobre datos reales (2015–2018).")

# ── Carga de artefactos ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = tf.keras.models.load_model("model/lstm_final.keras")
    scaler  = joblib.load("model/scaler.pkl")
    features  = joblib.load("model/features.pkl")
    target_idx = joblib.load("model/target_idx.pkl")
    window  = joblib.load("model/window.pkl")
    return model, scaler, features, target_idx, window

@st.cache_data
def load_data():
    energy  = pd.read_csv("data/energy_dataset.csv")
    weather = pd.read_csv("data/weather_features.csv")

    energy["time"]    = pd.to_datetime(energy["time"], utc=True)
    weather["dt_iso"] = pd.to_datetime(weather["dt_iso"], utc=True)

    # Columnas vacías
    cols_drop = ["generation hydro pumped storage aggregated", "forecast wind offshore eday ahead"]
    energy.drop(columns=[c for c in cols_drop if c in energy.columns], inplace=True)
    energy.dropna(subset=["total load actual"], inplace=True)
    energy.interpolate(method="linear", inplace=True)

    # Agregado clima
    weather_agg = weather.groupby("dt_iso").agg({
        "temp": "mean", "humidity": "mean",
        "wind_speed": "mean", "clouds_all": "mean"
    }).reset_index()
    weather_agg.columns = ["time", "temp", "humidity", "wind_speed", "clouds"]

    df = pd.merge(energy, weather_agg, on="time", how="inner")
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Features temporales cíclicas
    df["hour_sin"]   = np.sin(2 * np.pi * df["time"].dt.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["time"].dt.hour / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["time"].dt.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["time"].dt.dayofweek / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["time"].dt.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["time"].dt.month / 12)

    return df

# ── Carga ────────────────────────────────────────────────────────────────────
try:
    model, scaler, FEATURES, TARGET_IDX, WINDOW = load_artifacts()
    df = load_data()
except Exception as e:
    st.error(f"Error cargando los artefactos: {e}")
    st.stop()

# ── Selector de fecha y hora ─────────────────────────────────────────────────
st.divider()
st.subheader("Selecciona fecha y hora")

min_date = df["time"].dt.date.min() + timedelta(days=1)
max_date = df["time"].dt.date.max()

col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("Fecha", value=min_date, min_value=min_date, max_value=max_date)
with col2:
    selected_hour = st.slider("Hora", min_value=0, max_value=23, value=12)

# Construimos el timestamp con UTC
selected_dt = pd.Timestamp(datetime(selected_date.year, selected_date.month, selected_date.day, selected_hour), tz="UTC")

# ── Predicción ───────────────────────────────────────────────────────────────
if st.button("Predecir consumo", type="primary"):

    # Buscamos el índice más cercano al timestamp seleccionado
    idx = df[df["time"] <= selected_dt].index
    if len(idx) < WINDOW:
        st.warning("No hay suficientes datos anteriores para esta fecha. Selecciona una fecha posterior.")
        st.stop()

    idx_target = idx[-1]

    # Ventana de 24 horas anteriores
    window_df = df.loc[idx_target - WINDOW + 1: idx_target, FEATURES]
    if len(window_df) < WINDOW:
        st.warning("Ventana incompleta. Prueba con otra fecha.")
        st.stop()

    # Normalización y predicción
    window_scaled = scaler.transform(window_df.values)
    X = window_scaled[np.newaxis, :, :]  # shape (1, 24, n_features)
    pred_scaled = model.predict(X, verbose=0)[0, 0]

    # Desnormalización
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, TARGET_IDX] = pred_scaled
    pred_mw = scaler.inverse_transform(dummy)[0, TARGET_IDX]

    # Valor real
    real_mw = df.loc[idx_target, "total load actual"]
    error_mw = abs(pred_mw - real_mw)
    error_pct = error_mw / real_mw * 100

    # ── Métricas ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Resultado para {selected_date} a las {selected_hour:02d}:00h")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicción LSTM", f"{pred_mw:,.0f} MW")
    m2.metric("Consumo real", f"{real_mw:,.0f} MW")
    m3.metric("Error absoluto", f"{error_mw:,.0f} MW", delta=f"{error_pct:.1f}%", delta_color="inverse")

    # ── Gráfica de contexto ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Contexto: últimas 24 horas")

    context = df.loc[idx_target - WINDOW + 1: idx_target + 1, ["time", "total load actual"]].copy()
    context_real = context["total load actual"].values
    context_time = context["time"].dt.strftime("%H:%M").values

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(WINDOW), context_real[:WINDOW], color="steelblue", linewidth=2, label="Consumo real (contexto)")
    ax.scatter(WINDOW, real_mw, color="steelblue", s=80, zorder=5)
    ax.scatter(WINDOW, pred_mw, color="coral", s=120, marker="*", zorder=6, label=f"Predicción: {pred_mw:,.0f} MW")
    ax.axvline(WINDOW - 1, color="gray", linestyle="--", linewidth=0.8)

    xticks = list(range(0, WINDOW + 1, 4))
    ax.set_xticks(xticks)
    ax.set_xticklabels([context_time[i] if i < WINDOW else f"{selected_hour:02d}:00" for i in xticks])
    ax.set_ylabel("MW")
    ax.set_title("Consumo real (azul) y predicción LSTM (estrella naranja)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # ── Previsión TSO ─────────────────────────────────────────────────────────
    tso_mw = df.loc[idx_target, "total load forecast"]
    tso_error = abs(tso_mw - real_mw)

    st.divider()
    st.subheader("Comparativa con la previsión del TSO")

    t1, t2, t3 = st.columns(3)
    t1.metric("Previsión TSO", f"{tso_mw:,.0f} MW")
    t2.metric("Error TSO", f"{tso_error:,.0f} MW")
    t3.metric("Error LSTM", f"{error_mw:,.0f} MW")
