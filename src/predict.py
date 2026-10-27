
"""Predict a house price using the trained ML model."""

from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"


def predict_house(features):
    """Return the predicted median house value in US dollars."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run this first: "
            "python -m src.train"
        )

    saved_model = joblib.load(MODEL_PATH)

    # Keep the same feature order used during training.
    input_data = pd.DataFrame(
        [features],
        columns=saved_model["feature_names"]
    )

    prediction = saved_model["pipeline"].predict(input_data)[0]

    return float(prediction)


if __name__ == "__main__":

    # Example house details from the California dataset.
    house = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 30,
        "total_rooms": 1500,
        "total_bedrooms": 300,
        "population": 800,
        "households": 280,
        "median_income": 5.0,
        "ocean_proximity": "NEAR BAY"
    }

    try:
        predicted_price = predict_house(house)

        print("\nHouse Price Prediction")
        print("---------------------")
        print(f"Estimated median house value: ${predicted_price:,.2f}")
        print("Note: This is an educational estimate.")

    except FileNotFoundError as error:
        print(error)
