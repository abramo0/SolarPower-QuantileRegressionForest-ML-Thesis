
import datetime
import math
import os
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.utils import resample

X_MIN = 0
X_MAX = 1000
STEP = 50
CLIPPING_FACTOR = 2
N_SPLIT = 5
RANDOM_STATE = 42
TYPE_MODELS = ["LinearRegression", "PolynomialRegression", "DecisionTreeRegressor", "GradientBoostingRegressor",
               "RandomForestRegressor", "RandomForestQuantileRegressor"]
NAME_MODELS = ["linear", "polynomial", "decision_tree", "gradient_boosting", "random_forest", "random_forest_quantile"]
DATABASE_NAMES = ["linear.db", "polynomial.db", "decision_tree.db", "gradient_boosting", "random_forest.db",
                  "random_forest_quantile.db"]
SAMPLER_TYPE = 'TPESampler'
SCORING = make_scorer(mean_squared_error)
DIRECTION = 'minimize'
N_TRIALS = 30
DEGREE = 1





def get_model(trial, model_type, random_state=42):
    if model_type == 'LinearRegression':
        return LinearRegression(fit_intercept=False)

    elif model_type == 'PolynomialRegression':
        degree = trial.suggest_int('degree', 2, 5)
        return make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())

    elif model_type == 'DecisionTreeRegressor':
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 20)
        }
        return DecisionTreeRegressor(**params, random_state=random_state)

    elif model_type == 'GradientBoostingRegressor':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 2, 20)
        }
        return GradientBoostingRegressor(**params, random_state=random_state)

    elif model_type == 'RandomForestRegressor':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 2, 20)
        }
        return RandomForestRegressor(**params, random_state=random_state)

    elif model_type == 'RandomForestQuantileRegressor':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 200)
        }
        return RandomForestQuantileRegressor(**params, random_state=random_state)

    else:
        raise ValueError(f"Model type {model_type} is not supported")

#Function to calculate polynomial model
def identify_poly_model(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    return poly, coeffs


def data_analysis(X_dataset, Y_dataset, date_dataset, X_train, Y_train, date_train, X_test, Y_test, date_test, file_suffix):
    #function that allows the visualization of scatterplots and kdeplots of datasets, trainingsets and testsets
    
    def _display_data_scatterplot(graph_index, X_data, Y_data, title):  # function to display scatterplots
        plt.subplot(2, 3, graph_index)
        plt.scatter(X_data, Y_data, edgecolors='b', facecolors='none', linewidth=5, s=2)
        plt.xlabel('ghi forecast')
        plt.ylabel('power actual scaled')
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)


    def _display_data_kdeplot(graph_index, X_data, Y_data):
        plt.subplot(2, 3, graph_index)
        sns.kdeplot(x=X_data, y=Y_data, fill=True)
        plt.xlabel('ghi forecast')
        plt.ylabel('power actual scaled')
        plt.grid()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    def _display_monthly_data_distribution(graph_index, type_set, name_set):
        plt.subplot(1, 3, graph_index)
        type_set.plot(kind='bar', color='b')
        plt.title(f'Occurrences per Month {name_set}')
        plt.xlabel('Month')
        plt.ylabel('Number of occurrences')
        plt.xticks(ticks=range(12), labels=['Gen', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
        plt.ylim(0, max_value_month+200)
        plt.grid()

    def _display_hour_data_distribution(graph_index, type_set, name_set):
        plt.subplot(1, 3, graph_index)
        type_set.plot(kind='bar', color='b')
        plt.title(f'Occurrences per Hour {name_set}')
        plt.xlabel('Hour')
        plt.ylabel('Number of occurrences')
        plt.xticks(rotation=45)
        plt.xticks(ticks=range(len(type_set.index)), labels=[int(x) for x in type_set.index], rotation=45)
        plt.ylim(0, max_value_hour+200)
        plt.grid()





    x_min_base = min(X_dataset['ghi_forecast'].min(), X_train[:, 0].min(), X_test[:, 0].min())
    x_max_base = max(X_dataset['ghi_forecast'].max(), X_train[:, 0].max(), X_test[:, 0].max())
    y_min_base = min(Y_dataset['power_actual_scaled'].min(), Y_train.min(), Y_test.min())
    y_max_base = max(Y_dataset['power_actual_scaled'].max(), Y_train.max(), Y_test.max())
    
    x_margin = 0.2 * (x_max_base - x_min_base)
    y_margin = 0.2 * (y_max_base - y_min_base)
    x_min = x_min_base - x_margin
    x_max = x_max_base + x_margin
    y_min = y_min_base - y_margin
    y_max = y_max_base + y_margin


    plt.figure(figsize=(18, 12))
    plt.suptitle("Data analysis", fontsize=20)
    _display_data_scatterplot(1, X_data=X_dataset['ghi_forecast'], Y_data=Y_dataset['power_actual_scaled'], title='Dataset')
    _display_data_scatterplot(2, X_data=X_train[:, 0], Y_data=Y_train, title='Trainingset')
    _display_data_scatterplot(3, X_data=X_test[:, 0], Y_data=Y_test, title='Testset')

    _display_data_kdeplot(graph_index=4, X_data=X_dataset['ghi_forecast'], Y_data=Y_dataset['power_actual_scaled'])
    _display_data_kdeplot(graph_index=5, X_data=X_train[:, 0], Y_data=Y_train)
    _display_data_kdeplot(graph_index=6, X_data=X_test[:, 0], Y_data=Y_test)
    
    plt.savefig(f'data_analysis_{file_suffix}.jpg')
    plt.show()


    plt.figure(figsize=(18, 12))
    plt.suptitle("Correlation matrix", fontsize=20)
    correlation_matrix = pd.concat([X_dataset, Y_dataset], axis=1).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.savefig(f'correlation_matrix_{file_suffix}.jpg')

    plt.show()
    
    
    #Analyze the distribution of data according to months and hours
    month_dataset = dataset.groupby(date_dataset.dt.month).size()
    month_trainingset = dataset.groupby(date_train.dt.month).size()
    month_testset = dataset.groupby(date_test.dt.month).size()
    max_value_month = max(month_dataset.max(), month_trainingset.max(), month_testset.max())

    plt.figure(figsize=(18, 12))
    plt.suptitle("Monthly data distribution", fontsize=20)
    _display_monthly_data_distribution(1, month_dataset, "dataset")
    _display_monthly_data_distribution(2, month_trainingset, "trainingset")
    _display_monthly_data_distribution(3, month_testset, "testset")
    plt.savefig(f'monthly_distribution_{file_suffix}.jpg')
    plt.show()

    hour_dataset = dataset.groupby(date_dataset.dt.hour).size()
    hour_trainingset = dataset.groupby(date_train.dt.hour).size()
    hour_testset = dataset.groupby(date_test.dt.hour).size()
    max_value_hour = max(hour_dataset.max(), hour_trainingset.max(), hour_testset.max())
    
    plt.figure(figsize=(18, 12))
    plt.suptitle("Hours data distribution", fontsize=20)
    _display_hour_data_distribution(1, hour_dataset, "dataset")
    _display_hour_data_distribution(2, hour_trainingset, "trainingset")
    _display_hour_data_distribution(3, hour_testset, "testset")
    plt.savefig(f'hour_distribution_{file_suffix}.jpg')
    plt.show()











def remove_outliers(df, target_variable, regressor, x_min, x_max, step, k):
    #removing outliers using iqr

    def _identify_outliers_iqr(data, k):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        is_outlier = ((data < (Q1 - k * IQR)) | (data > (Q3 + k * IQR)))
        return is_outlier

    outliers_indexes_list = []
    for i in np.arange(start=x_min, stop=x_max, step=step):
        lower_bound = i
        upper_bound = i + step

        data_interval = df[target_variable][((df[regressor] < upper_bound) &
                                             (df[regressor] >= lower_bound))]

        outliers_indexes_list.append(_identify_outliers_iqr(data_interval, k))

    outliers_indexes = pd.concat(outliers_indexes_list).sort_index()
    df_clean = df.loc[~outliers_indexes.values]

    return df_clean



def objective(trial, model_type, scoring, X_train, y_train, n_splits):
    #function used for hyperparameter optimization
    model = get_model(trial, model_type)
    score = cross_val_score(model, X_train, y_train, cv=TimeSeriesSplit(n_splits=n_splits), scoring=scoring)
    return np.mean(score)  #returns the average of the scores obtained from cross-validation



def study_model(X_train, y_train, X_test, type_model, name_model, n_splits, score, direction, sampler_type, db_file, n_trials, random_state=42):  # carries out the study for each model

    if sampler_type == 'BruteForceSampler':
        sampler = optuna.samplers.BruteForceSampler(seed=random_state)
    elif sampler_type == 'RandomSampler':
        sampler = optuna.samplers.RandomSampler(seed=random_state)
    elif sampler_type == 'TPESampler':
        sampler = optuna.samplers.TPESampler(seed=random_state)
    else:
        raise ValueError(f"Sampler type {sampler_type} is not supported")

    # Create an Optuna study and optimize
    study_name = "ottimizzazione_" + name_model + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if os.path.exists(db_file):  # if the db already exists, delete it first
        os.remove(db_file)
    study = optuna.create_study(study_name=study_name, sampler=sampler, direction=direction, storage='sqlite:///study.db')
    study.optimize(lambda trial: objective(trial, type_model, score, X_train, y_train, n_splits), n_trials=n_trials)

    #Train the final model with the best hyperparameters found
    best_params = study.best_params
    best_model = get_model(optuna.trial.FixedTrial(best_params), type_model, random_state=random_state)
    best_model.fit(X_train, y_train)
    if type_model=="RandomForestQuantileRegressor":
        Y_train_pred = best_model.predict(X_train, quantiles="mean")
        Y_test_pred = best_model.predict(X_test, quantiles="mean")
    else:
        Y_train_pred = best_model.predict(X_train)
        Y_test_pred = best_model.predict(X_test)

    # Prepare results string
    if type_model == 'PolynomialRegression':
        polynomial_coefficients = None
        polynomial_coefficients = best_model.named_steps['linearregression'].coef_
        results_string = "=" * 50 + f"\nModel: {name_model}\nBest hyperparameters: {best_params}\nBest scoring: {study.best_value:.4f}\nPolynomial Coefficients: {polynomial_coefficients}\n"
    else:
        results_string = "=" * 50 + f"\nModel: {name_model}\nBest hyperparameters: {best_params}\nBest scoring: {study.best_value:.4f}\n"

    def _save_data(file_path, results_string):
        if type_model == "LinearRegression":
            with open(file_path, "w") as file:
                file.write(results_string)
        else:
            with open(file_path, "a") as file:
                file.write(results_string)
        
    _save_data(file_path="data_results.txt", results_string=results_string)
    print(results_string)

    return best_model, Y_train_pred, Y_test_pred, best_params






def compare_models(variables, X_train, Y_train, X_test, Y_test, name_models, degree, n_splits):
    #compare the results obtained by each model
    
    x_poly = np.linspace(0, 0.9, 500)
    for i in range(len(name_models)):
        Y_test_pred = variables[f"Y_test_pred_{name_models[i]}"]
        # Set up the figure and subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 18))
        fig.suptitle(f'Evaluation Plots for {name_models[i]}', fontsize=16)

        y_test_pred = variables[f"Y_test_pred_{name_models[i]}"]
        poly_test, _ = identify_poly_model(y_test_pred, Y_test, degree)

        # First subplot: Scatter Plot of predicted values and real values
        ax1 = axes[0, 0]
        ax1.scatter(Y_test_pred, Y_test, edgecolors='blue', facecolors='none', linewidths=2, s=10)
        ax1.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], '--', lw=3, color='black', label="Identity line", zorder=3)
        ax1.plot(x_poly, poly_test(x_poly), linewidth=4, label=f'Linear regression', c='tab:red', zorder=2)
        ax1.set_xlabel('Predicted power')
        ax1.set_ylabel('Real power')
        ax1.set_title('Scatterplot of predicted vs real power (test data)')
        ax1.grid(True)
        ax1.legend().set_draggable(True)

        # Second subplot: Residual plot
        residuals = Y_test - Y_test_pred
        ax2 = axes[0, 1]
        ax2.scatter(Y_test_pred, residuals, edgecolors='blue', facecolors='none', linewidths=2, s=10)
        ax2.plot([min(Y_test_pred), max(Y_test_pred)], [0, 0], color='black', linestyle='--', lw=3)
        ax2.set_title('Residual plot on test data')
        ax2.set_xlabel('Predicted power')
        ax2.set_ylabel('Residuals')
        ax2.grid(True)

        # Third subplot: Distribution of errors
        ax3 = axes[1, 0]
        sns.histplot(residuals, kde=True, color='blue', ax=ax3)
        ax3.set_title('Distribution of test Errors')
        ax3.set_xlabel('Errors')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)

        # Fourth subplot: Learning curve with TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        train_sizes, train_scores, test_scores = learning_curve(variables[f"model_{name_models[i]}"], X_train, Y_train, cv=tscv)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        ax4 = axes[1, 1]
        ax4.plot(train_sizes, train_scores_mean, label='Training score')
        ax4.plot(train_sizes, test_scores_mean, label='Cross-validation score')
        ax4.set_xlabel('Number of samples in training set')
        ax4.set_ylabel('Score')
        ax4.set_title(f'Learning Curves for {name_models[i]}')
        ax4.legend().set_draggable(True)
        ax4.grid(True)

        plt.savefig(f'model_{name_models[i]}.jpg')
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.show()


    #comparison of r^2 and MSE values
    plt.figure(figsize=(18, 12))
    plt.suptitle('Comparison of the $R^2$ and MSE parameters', fontsize=20)
    plt.subplot(1, 2, 1)
    for i in range(len(name_models)):
        y_test_pred = variables[f"Y_test_pred_{name_models[i]}"]
        values = r2_score(Y_test, y_test_pred)
        plt.bar(name_models[i], values, label=name_models[i])
        plt.text(i, values, f'$R^2$: {round(values, 3)}', ha='center', va='bottom')
    plt.xticks([])
    plt.xlabel('Model')
    plt.ylabel('Value of $R^2$')
    plt.title('Comparison of $R^2$ values in test')
    plt.grid()
    plt.legend(loc="lower right", bbox_to_anchor=(1, 0)).set_draggable(True)

    plt.subplot(1, 2, 2)
    for i in range(len(name_models)):
        y_test_pred = variables[f"Y_test_pred_{name_models[i]}"]
        values = mean_squared_error(Y_test, y_test_pred)
        plt.bar(name_models[i], values, label=name_models[i])
        plt.text(i, values, f'MSE: {round(values, 3)}', ha='center', va='bottom')
    plt.xticks([])
    plt.xlabel('Model')
    plt.ylabel('Value of MSE')
    plt.title('Comparison of MSE values in test')
    plt.grid()
    plt.legend(loc="lower right", bbox_to_anchor=(1, 0)).set_draggable(True)
    plt.savefig(f'comparison_r2_mse.jpg')
    plt.show()

    # scatterplot train values
    plt.figure(figsize=(18, 12))
    plt.suptitle('Scatterplot of predicted power vs real power on train values', fontsize=20)
    for i in range(1, len(name_models) + 1):
        plt.subplot(2, math.ceil(len(name_models) / 2), i)
        i = i - 1
        # Creating polynomial models for bias testing on gof
        y_train_pred = variables[f"Y_train_pred_{name_models[i]}"]
        poly_train, _ = identify_poly_model(y_train_pred, Y_train, degree)

        plt.scatter(variables[f"Y_train_pred_{name_models[i]}"], Y_train, linewidths=7, label=name_models[i],
                    facecolors='none', edgecolor='b', s=7)
        plt.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], '--', lw=3, color='black', zorder=3, label="Identity line")
        plt.plot(x_poly, poly_train(x_poly), linewidth=4, label=f'Linear regression',
                     c='tab:red', zorder=2)

        plt.xlabel('Predict power')
        plt.ylabel('Real power')
        plt.legend().set_draggable(True)
        plt.grid()
    plt.savefig('GoF_training_values.jpg')
    plt.show()

    # scatterplot test values
    plt.figure(figsize=(18, 12))
    plt.suptitle('Scatterplot of predicted power vs actual power on test values', fontsize=20)
    for i in range(1, len(name_models) + 1):

        plt.subplot(2, math.ceil(len(name_models) / 2), i)
        i = i - 1
        y_test_pred = variables[f"Y_test_pred_{name_models[i]}"]
        poly_test, _ = identify_poly_model(y_test_pred, Y_test, degree)

        plt.scatter(variables[f"Y_test_pred_{name_models[i]}"], Y_test, linewidths=7, label=name_models[i],
                    facecolors='none', edgecolor='b', s=7)
        plt.plot(x_poly, poly_test(x_poly), linewidth=4, label=f'Linear regression',
                     c='tab:red', zorder=2)
        plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], '--', lw=3, color='black', zorder=3, label="Identity line")
        plt.xlabel('Real power')
        plt.ylabel('Predict power')
        plt.legend().set_draggable(True)
        plt.grid()
    plt.savefig('GoF_test_values.jpg')
    plt.show()

    #models applied on power vs ghi graph in the case of training and test
    plt.figure(figsize=(18, 12))
    plt.suptitle('Models on power vs GHI forecast', fontsize=20)
    for i in range(1, len(name_models) + 1):
        plt.subplot(2, len(name_models), i)
        i = i - 1
        X_train_array = X_train.flatten()
        sorted_indices = np.argsort(X_train_array)
        sorted_X_train = X_train_array[sorted_indices]
        sorted_Y_train_pred = variables[f"Y_train_pred_{name_models[i]}"][sorted_indices]
        plt.plot(sorted_X_train, sorted_Y_train_pred, label=name_models[i], linewidth=4, color='r')
        plt.scatter(X_train, Y_train, label='Training data', zorder=0, color='b')
        plt.title('Training')
        plt.xlabel('GHI forecast')
        plt.ylabel('Power')
        plt.legend().set_draggable(True)
        plt.grid()
    for i in range(1, len(name_models) + 1):
        plt.subplot(2, len(name_models), len(name_models) + i)
        i = i - 1
        X_test_array = X_test.flatten()
        sorted_indices = np.argsort(X_test_array)
        sorted_X_test = X_test_array[sorted_indices]
        sorted_Y_test_pred = variables[f"Y_test_pred_{name_models[i]}"][sorted_indices]
        plt.plot(sorted_X_test, sorted_Y_test_pred, label=name_models[i], linewidth=4, color='r')
        plt.scatter(X_test, Y_test, label='Test data', zorder=0, color='b')
        plt.title('Test')
        plt.xlabel('GHI forecast')
        plt.ylabel('Power')
        plt.legend().set_draggable(True)
        plt.grid()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.savefig('power_ghi_training_test.jpg')
    plt.show()


def study_with_bins(model_variables, dataset, X_train, name_models, first_quantile, second_quantile):
    # Divisione in bin e applicazione del modello su media e mediana della potenza
    bins = [0, 50, 100, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    dataset['GHI_bins'] = pd.cut(dataset['ghi_forecast'], bins=bins)
    
    # Calcola la media, la mediana, il 10° quantile e il 90° quantile
    mean_power_by_bin = dataset.groupby('GHI_bins', observed=False)['power_actual_scaled'].mean()
    median_power_by_bin = dataset.groupby('GHI_bins', observed=False)['power_actual_scaled'].median()
    q10_power_by_bin = dataset.groupby('GHI_bins', observed=False)['power_actual_scaled'].quantile(first_quantile)
    q90_power_by_bin = dataset.groupby('GHI_bins', observed=False)['power_actual_scaled'].quantile(second_quantile)

    plt.figure(figsize=(18, 12))
    plt.suptitle('Mean, Median, and Quantiles', fontsize=20)

    for i in range(1, len(name_models) + 1):
        plt.subplot(math.ceil(len(name_models) / 3), 3, i)
        i = i - 1
        X_train_array = X_train.flatten()
        sorted_indices = np.argsort(X_train_array)
        sorted_X_train = X_train_array[sorted_indices]
        sorted_Y_train_pred = model_variables[f"Y_train_pred_{name_models[i]}"][sorted_indices]
        
        # Tracciare le curve predette
        plt.plot(sorted_X_train, sorted_Y_train_pred, linewidth=2, label=name_models[i], zorder=0)
        
        # Aggiungere i punti della media, mediana, quantile 10% e quantile 90%
        plt.scatter(mean_power_by_bin.index.categories.mid, mean_power_by_bin, label='Mean Power', color="red")
        plt.scatter(median_power_by_bin.index.categories.mid, median_power_by_bin, label='Median Power', color="green")
        plt.plot(q10_power_by_bin.index.categories.mid, q10_power_by_bin, label=f'{first_quantile*100:.0f}% Quantile', color="blue")
        plt.plot(q90_power_by_bin.index.categories.mid, q90_power_by_bin, label=f'{second_quantile*100:.0f}% Quantile', color="purple")
        
        plt.xlabel('GHI forecast')
        plt.ylabel('Power')
        plt.grid()
        plt.legend().set_draggable(True)
    
    # Salva e mostra il grafico
    plt.savefig('analysis_mean_median_quantiles_with_bins_.jpg')
    plt.show()


if __name__=="__main__":
    #Data files
    dataset_file = r"pv_dataset.csv"
    dataset = pd.read_csv(dataset_file)
    
    dataset = dataset[['date', 'power_actual_scaled', 'ghi_forecast']]
    dataset.dropna(inplace=True)  #Remove rows with NaN
    date_dataset = pd.to_datetime(dataset['date'])
    X_dataset = dataset[['ghi_forecast']]
    Y_dataset = dataset[['power_actual_scaled']]


    trainingset_file = "pv_trainingset.csv"
    trainingset = pd.read_csv(trainingset_file)
    trainingset.dropna(inplace=True)
    date_traingset=pd.to_datetime(trainingset['date'])
    X_train = trainingset['ghi_forecast'].values.reshape(-1, 1)
    Y_train = trainingset['power_actual_scaled'].values.ravel()

    testset_file = "pv_testset.csv"
    testset = pd.read_csv(testset_file)
    testset.dropna(inplace=True)
    date_testset=pd.to_datetime(testset['date'])
    X_test = testset['ghi_forecast'].values.reshape(-1, 1)
    Y_test = testset['power_actual_scaled'].values.ravel()

    data = {
        'timestamp': pd.date_range(start='2021-01-02', periods=1000, freq='H'),
        'ghi': abs(1000 * pd.np.sin(pd.np.linspace(0, 50, 1000))),
        'power': abs(500 * pd.np.sin(pd.np.linspace(0, 50, 1000)) + 200)
    }
    df = pd.DataFrame(data)

    # Converti il timestamp in datetime (se necessario)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filtra una settimana (dal 1 gennaio 2024 al 7 gennaio 2024)
    start_date = '2021-01-02'
    end_date = '2021-01-08'
    df_week = df[(df['timestamp'] >= start_date) & (df['timestamp'] < end_date)]

    # Plot dei dati
    plt.figure(figsize=(15, 6))

    plt.plot(df_week['timestamp'], df_week['ghi'], label='GHI', color='orange')
    plt.plot(df_week['timestamp'], df_week['power'], label='Power', color='blue')

    plt.title('Serie Temporali di GHI e Power (1 settimana)')
    plt.xlabel('Timestamp')
    plt.ylabel('Valore')
    plt.legend()
    plt.grid()
    plt.show()





    data_analysis(X_dataset=X_dataset, Y_dataset=Y_dataset, date_dataset=date_dataset, X_train=X_train, Y_train=Y_train, date_train=date_traingset, X_test=X_test, Y_test=Y_test, date_test=date_testset, file_suffix="with_outliers")

    trainingset_clean = remove_outliers(trainingset, 'power_actual_scaled', 'ghi_forecast', X_MIN, X_MAX, STEP, CLIPPING_FACTOR)

    X_train_clean = trainingset_clean['ghi_forecast'].values.reshape(-1, 1)
    Y_train_clean = trainingset_clean['power_actual_scaled'].values.ravel()

    data_analysis(X_dataset=X_dataset, Y_dataset=Y_dataset, date_dataset=date_dataset, X_train=X_train_clean, Y_train=Y_train_clean, date_train=date_traingset, X_test=X_test, Y_test=Y_test, date_test=date_testset, file_suffix="without_outliers")

    variables = {}
    for i in range(len(TYPE_MODELS)):
        output = study_model(X_train, Y_train, X_test, TYPE_MODELS[i],
                             NAME_MODELS[i], N_SPLIT, SCORING, DIRECTION,
                             SAMPLER_TYPE, DATABASE_NAMES[i], N_TRIALS, RANDOM_STATE)
        variables[f"model_{NAME_MODELS[i]}"], variables[f"Y_train_pred_{NAME_MODELS[i]}"], variables[
            f"Y_test_pred_{NAME_MODELS[i]}"], variables[f"best_params_{NAME_MODELS[i]}"] = output

    compare_models(variables, X_train, Y_train, X_test, Y_test, NAME_MODELS, DEGREE, N_SPLIT)
    study_with_bins(variables, dataset, X_train, NAME_MODELS, 0.10, 0.90)
