# House Price Prediction

An end-to-end beginner-friendly machine learning project that predicts California house values using scikit-learn. Includes data exploration, preprocessing, model training/evaluation, prediction CLI, and an optional Streamlit app.

## Features
- Loads the California Housing dataset through scikit-learn (internet may be required on first run).
- Splits data into training and test sets.
- Trains Linear Regression, Decision Tree, and Random Forest models.
- Compares models using MAE, RMSE, and R².
- Saves the best model locally under `models/` and metrics under `reports/`.
- Provides a Streamlit form to estimate house value from numeric features.
- Includes a notebook for exploratory data analysis.

> Dataset note: California Housing is a built-in scikit-learn dataset originally derived from the 1990 U.S. Census. The target is median house value in units of $100,000. It is for learning only and is not a reliable real-world property valuation tool.

## Project structure
```text
house-price-prediction/
├── data/
│   ├── raw/                 # Optional local data; not tracked
│   └── processed/            # Optional processed data; not tracked
├── notebooks/
│   └── house_price_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train.py
│   └── predict.py
├── models/                   # Generated model files; not tracked
├── reports/                  # Generated metrics; not tracked
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup in VS Code (Windows PowerShell)
1. Open this folder in VS Code.
2. Open Terminal → New Terminal.
3. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If activation is blocked, use Command Prompt and run `.venv\Scripts\activate.bat`.
4. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
5. Train and evaluate:
   ```powershell
   python -m src.train
   ```
6. Make a sample prediction:
   ```powershell
   python -m src.predict
   ```
7. Launch the web app:
   ```powershell
   streamlit run app.py
   ```

## Input features
- `MedInc`: median income in block group (in tens of thousands of dollars)
- `HouseAge`: median house age
- `AveRooms`: average rooms per household
- `AveBedrms`: average bedrooms per household
- `Population`: block-group population
- `AveOccup`: average household occupancy
- `Latitude`: latitude
- `Longitude`: longitude

## Generated outputs
After training, these files are created locally:
- `models/best_model.joblib`
- `reports/model_metrics.csv`
- `reports/actual_vs_predicted.png`

They are intentionally excluded from Git by default. If you want to include a small model artifact in your portfolio, review its size and repository policy before changing `.gitignore`.

## Limitations
- This example uses California data, not Indian property data.
- Model estimates are educational approximations and should not be used for financial or property decisions.
- A model's test metrics do not guarantee accuracy for different regions or future markets.

## GitHub
Create a GitHub repository named `house-price-prediction`, then from the project root:
```bash
git init
git add .
git commit -m "Add house price prediction ML project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/house-price-prediction.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username.
