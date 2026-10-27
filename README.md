# House Price Prediction Using Machine Learning

## Project Overview

This project predicts house values in California using machine learning algorithms. It demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, model evaluation, and an interactive web application built with Streamlit.

## Objectives

- Clean and preprocess housing data.
- Perform exploratory data analysis (EDA).
- Train and compare multiple regression algorithms.
- Evaluate models using standard regression metrics.
- Build an interactive application for house-price predictions.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Web Application:** Streamlit
- **Model Storage:** Joblib
- **Version Control:** Git and GitHub

## Machine Learning Algorithms

The project compares the following algorithms:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

The best-performing model is selected based on the evaluation approach implemented in the training script.

## Dataset

The project uses the California housing dataset, containing features such as longitude, latitude, housing median age, total rooms, total bedrooms, population, households, median income, and ocean proximity.

The target variable is `median_house_value`.

**Dataset setup:** Download the compatible California housing `housing.csv` dataset and place it at `data/raw/housing.csv`. The dataset is excluded from Git by `.gitignore`, so it must be downloaded separately.

## Project Structure

```text
house-price-prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
└── src/
    ├── data_preprocessing.py
    ├── train.py
    └── predict.py
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Praveen-1131/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place `housing.csv` inside `data/raw/`.

### 5. Train the models

Run the training script from the project root:

```bash
python src/train.py
```

### 6. Launch the web application

```bash
python -m streamlit run app.py
```

Enter the housing details in the application to generate a prediction.

## Model Evaluation

The project evaluates regression models using metrics such as:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

Refer to the generated evaluation report for the actual model results.

## Key Learning Outcomes

- Data cleaning and preprocessing
- Handling categorical variables
- Building and comparing regression models
- Evaluating machine learning performance
- Saving and loading trained models
- Developing an interactive Streamlit application
- Managing a project with Git and GitHub

## Future Improvements

- Add more detailed EDA visualizations.
- Tune model hyperparameters.
- Deploy the Streamlit application online.
- Add automated testing and continuous integration.

## Author

**Praveen**

GitHub: https://github.com/Praveen-1131
