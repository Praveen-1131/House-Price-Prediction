
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write(
    "Enter housing details to estimate the median house value "
    "using a machine learning model trained on California data."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Train the model first.")
    st.code("python -m src.train")
    st.stop()

# Load the model saved by src/train.py
saved_model = joblib.load(MODEL_PATH)
pipeline = saved_model["pipeline"]
feature_names = saved_model["feature_names"]

st.subheader("Enter House Details")

with st.form("house_form"):
    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input(
            "Longitude", value=-122.23, step=0.01
        )
        latitude = st.number_input(
            "Latitude", value=37.88, step=0.01
        )
        age = st.number_input(
            "Housing median age", min_value=1.0, value=30.0
        )
        rooms = st.number_input(
            "Total rooms", min_value=1.0, value=1500.0
        )
        bedrooms = st.number_input(
            "Total bedrooms", min_value=0.0, value=300.0
        )

    with col2:
        population = st.number_input(
            "Population", min_value=1.0, value=800.0
        )
        households = st.number_input(
            "Households", min_value=1.0, value=280.0
        )
        income = st.number_input(
            "Median income", min_value=0.0, value=5.0
        )
        proximity = st.selectbox(
            "Ocean proximity",
            ["INLAND", "NEAR BAY", "NEAR OCEAN",
             "ISLAND", "<1H OCEAN"]
        )

    submitted = st.form_submit_button("Predict House Price")

if submitted:
    house = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": age,
        "total_rooms": rooms,
        "total_bedrooms": bedrooms,
        "population": population,
        "households": households,
        "median_income": income,
        "ocean_proximity": proximity,
    }

    try:
        input_df = pd.DataFrame(
            [house], columns=feature_names
        )
        prediction = float(pipeline.predict(input_df)[0])

        st.success("Prediction completed!")
        st.metric(
            "Estimated Median House Value",
            f"${prediction:,.2f}"
        )
        st.caption(
            "Educational estimate only; this model uses "
            "California housing data, not Indian property data."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
