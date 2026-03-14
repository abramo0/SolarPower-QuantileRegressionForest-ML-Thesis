# PV Power Prediction using Machine Learning

This repository contains a **photovoltaic power prediction project** using various regression models, including **linear, polynomial, decision trees, and quantile random forests**.
The project is implemented in **Python** and leverages libraries such as `scikit-learn`, `optuna`, `pandas`, `numpy`, `matplotlib`, and `seaborn` for analysis, model optimization, and visualization.

---

## Repository Structure

project-root/  
│  
├── file/  
│   ├── pv_dataset_sample.csv  
│   ├── pv_trainingset.csv  
│   └── pv_testset.csv  
│  
├── images/  
│   ├── data_analysis_with_outliers.jpg  
│   ├── data_analysis_without_outliers.jpg  
│   ├── comparison_r2_mse.jpg  
│   └── ...  
├── results/  
│   └── results.txt  
│  
├── main.py  
├── requirements.txt  
└── README.md  


---

## Dataset

- **`pv_dataset.csv`**: dataset for testing the code  
- **`pv_trainingset.csv`** and **`pv_testset.csv`**: training and test subsets

---

## Installation

Clone the repository:

```bash
git clone https://github.com/abramo0/SolarPower-QuantileRegressionForest-ML-Thesis.git
cd SolarPower-QuantileRegressionForest-ML-Thesis
```

Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```
Install required libraries:
```
pip install -r requirements.txt
```

---
## Usage

Run the main script to perform analysis and model training:
```
python main.py
```

The script will:
1. Load the dataset (pv_dataset.csv or full CSV)
2. Perform data analysis and generate plots (results/data_analysis_*.jpg)
3. Remove outliers and generate updated plots
4. Study and optimize models using Optuna
5. Compare models (scatterplots, residuals, learning curves, R², and MSE)
6. Apply models to GHI bins for mean, median, and quantile analysis

---
## Results

All generated plots are saved in the results/ folder:
- data_analysis_with_outliers.jpg → Scatter and KDE plots of original data
- data_analysis_without_outliers.jpg → Data cleaned from outliers
- comparison_r2_mse.jpg → Model metric comparison
- model_<model_name>.jpg → Detailed evaluation of each model
- power_ghi_training_test.jpg → Models applied to GHI vs power

---
## Customization

Key parameters can be modified directly in main.py:


| Parameter       | Description                          | Default                                                                                                                                        |
| --------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| TYPE_MODELS     | Models to train                      | LinearRegression, PolynomialRegression, DecisionTreeRegressor, GradientBoostingRegressor, RandomForestRegressor, RandomForestQuantileRegressor |
| N_SPLIT         | Number of splits for TimeSeriesSplit | 5                                                                                                                                              |
| N_TRIALS        | Number of Optuna trials              | 30                                                                                                                                             |
| CLIPPING_FACTOR | Outlier removal factor               | 2                                                                                                                                              |
| DEGREE          | Degree for polynomial regression     | 1                                                                                                                                              |
| SCORING         | Scoring function for Optuna          | mean_squared_error                                                                                                                             |

---
## License

This project is licensed under the MIT License
