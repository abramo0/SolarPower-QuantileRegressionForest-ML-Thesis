PV Power Prediction using Machine Learning

This repository contains a photovoltaic power prediction project using various regression models, including linear, polynomial, decision trees, and quantile random forests. The project is implemented in Python and leverages libraries such as scikit-learn, optuna, pandas, numpy, matplotlib, and seaborn for analysis, model optimization, and visualization.

Repository Structure

project-root/
│
├── data/ # CSV datasets
│ ├── pv_dataset_sample.csv # Sample dataset (lightweight)
│ ├── pv_trainingset.csv
│ └── pv_testset.csv
│
├── models/ # Saved trained models (optional, .pkl)
│
├── results/ # Generated plots and results
│ ├── data_analysis_with_outliers.jpg
│ ├── data_analysis_without_outliers.jpg
│ ├── comparison_r2_mse.jpg
│ └── ...
│
├── main.py # Main script for training and analysis
├── requirements.txt # Required Python libraries
└── README.md # This file

Dataset

The main dataset pv_dataset.csv is too large to include in the repository. For testing purposes, a lightweight sample dataset is included (pv_dataset_sample.csv).

pv_dataset_sample.csv: sample dataset for testing the code.

pv_trainingset.csv and pv_testset.csv: training and test subsets.

For the full dataset (required for real results): download from the link provided by your organization or source. Save the full file as data/pv_dataset.csv.

Installation

Clone the repository:

git clone https://github.com/your-username/your-repo.git

cd your-repo

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate # Linux / Mac
venv\Scripts\activate # Windows

Install required libraries:

pip install -r requirements.txt

Example requirements.txt content:

pandas
numpy
scikit-learn
matplotlib
seaborn
optuna
quantile_forest

Usage

Run the main script to perform analysis and model training:

python main.py

The script will:

Load the dataset (pv_dataset_sample.csv or full CSV)

Perform data analysis and generate plots (results/data_analysis_*.jpg)

Remove outliers and generate updated plots

Study and optimize models using Optuna

Compare models (scatterplots, residuals, learning curves, R², and MSE)

Apply models to GHI bins for mean, median, and quantile analysis

Model Saving (Optional)

To save trained models for future use:

import pickle
with open("models/random_forest.pkl", "wb") as f:
  pickle.dump(best_model, f)

Load a saved model later:

with open("models/random_forest.pkl", "rb") as f:
  model = pickle.load(f)

Results

All generated plots are saved in the results/ folder:

data_analysis_with_outliers.jpg → Scatter and KDE plots of original data

data_analysis_without_outliers.jpg → Data cleaned from outliers

comparison_r2_mse.jpg → Model metric comparison

model_<model_name>.jpg → Detailed evaluation of each model

power_ghi_training_test.jpg → Models applied to GHI vs power

Customization

Key parameters can be modified directly in main.py:

Parameter	Description	Default
TYPE_MODELS	Models to train	LinearRegression, PolynomialRegression, DecisionTreeRegressor, GradientBoostingRegressor, RandomForestRegressor, RandomForestQuantileRegressor
N_SPLIT	Number of splits for TimeSeriesSplit	5
N_TRIALS	Number of Optuna trials	30
CLIPPING_FACTOR	Outlier removal factor	2
DEGREE	Degree for polynomial regression	1
SCORING	Scoring function for Optuna	mean_squared_error
Notes

The full CSV is not included due to size

Generated plots can be deleted or regenerated at any time

Saved models (.pkl) are optional and only used to avoid retraining

License

This project is licensed under the MIT License
