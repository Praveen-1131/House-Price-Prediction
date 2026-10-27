
from pathlib import Path
import csv
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import load_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test, preprocessor, features = load_data()

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=14, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    best_rmse = float("inf")
    best_pipeline = None
    best_name = ""
    best_predictions = None

    for name, model in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = float(np.sqrt(
            mean_squared_error(y_test, predictions)
        ))
        r2 = r2_score(y_test, predictions)

        print(f"MAE:  ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R2:   {r2:.4f}")

        results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_pipeline = pipeline
            best_name = name
            best_predictions = predictions

    joblib.dump({
        "pipeline": best_pipeline,
        "model_name": best_name,
        "feature_names": features
    }, MODEL_DIR / "best_model.joblib")

    with open(
        REPORT_DIR / "model_metrics.csv",
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["model", "MAE", "RMSE", "R2"]
        )
        writer.writeheader()
        writer.writerows(results)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, best_predictions, alpha=0.25)

    low = min(y_test.min(), best_predictions.min())
    high = max(y_test.max(), best_predictions.max())

    plt.plot([low, high], [low, high], "--")
    plt.xlabel("Actual house value ($)")
    plt.ylabel("Predicted house value ($)")
    plt.title(f"Actual vs Predicted: {best_name}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "actual_vs_predicted.png")
    plt.close()

    print(f"\nBest model: {best_name}")
    print("Model and reports saved successfully.")


if __name__ == "__main__":
    main()
