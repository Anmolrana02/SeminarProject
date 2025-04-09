# 📊 Efficient Irrigation Electricity Predictor – Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load model or train quickly (simple demo)
@st.cache_resource
def load_model():
    df = pd.read_csv("Western_UP_5_Districts_Irrigation_Data.csv")
    df["crop"] = LabelEncoder().fit_transform(df["crop"])
    df["district"] = LabelEncoder().fit_transform(df["district"])
    df["month"] = pd.to_datetime(df["date"]).dt.month

    features = ["temperature_C", "precip_mm", "wind_m_s", "solar_rad_MJ_m2",
                "soil_moisture", "crop", "district", "month"]
    X = df[features]
    y = df["electricity_kWh"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, df

model, df = load_model()

# UI
st.title("⚡ Efficient Electricity Use in Irrigation – Predictor")
st.markdown("Predict electricity usage (kWh) based on field & weather conditions")

# Input widgets
district = st.selectbox("District", df["district"].unique())
crop = st.selectbox("Crop", df["crop"].unique())
month = st.slider("Month", 1, 12, 6)

col1, col2 = st.columns(2)

with col1:
    temp = st.number_input("Temperature (°C)", 10.0, 45.0, 30.0)
    solar = st.number_input("Solar Radiation (MJ/m²)", 5.0, 30.0, 18.0)

with col2:
    precip = st.number_input("Rainfall (mm)", 0.0, 50.0, 5.0)
    soil_moisture = st.number_input("Soil Moisture (%)", 10.0, 60.0, 25.0)

wind = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0)

# Prediction
input_df = pd.DataFrame([{
    "temperature_C": temp,
    "precip_mm": precip,
    "wind_m_s": wind,
    "solar_rad_MJ_m2": solar,
    "soil_moisture": soil_moisture,
    "crop": crop,
    "district": district,
    "month": month
}])

if st.button("🔮 Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Electricity Usage: **{prediction:.2f} kWh**")

# Optional: Plot historical avg usage
if st.checkbox("📊 Show average electricity by month"):
    st.line_chart(df.groupby("month")["electricity_kWh"].mean())
