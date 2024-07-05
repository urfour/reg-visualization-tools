from typing import Union
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from math import ceil
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_dataset(data : Path, sep : str, save_dir : Path, target : str, categorical_features : list, to_drop : list):
    """ Train a dataset with two models: XGBoost and DecisionTreeRegressor.
    The results are saved in a csv file in the results folder.

    Parameters:
    data (str): The path to the dataset.
    sep (str): The separator for the dataset.
    save_dir (str or Path): The path to save the results.
    target (str): The target column.
    categorical_features (list): The categorical features to be one-hot encoded.
    to_drop (list): The columns to be dropped.
    """
    df = pd.read_csv(data, sep=sep)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame()
    metrics = {}
    metrics_df = pd.DataFrame()

    if to_drop is not None:
        df = df.drop(columns=to_drop)
    df = pd.get_dummies(df, columns=categorical_features)

    x = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model1 = XGBRegressor()
    model1.fit(X_train, y_train)
    model2 = DecisionTreeRegressor()
    model2.fit(X_train, y_train)

    pred_1 = model1.predict(X_test)
    pred_2 = model2.predict(X_test)

    results_df[target] = y_test
    results_df[target+'_model1'] = pred_1.round(0)
    results_df[target+'_model2'] = pred_2.round(0)
    results_df['error_model1'] = results_df[target] - results_df[target+'_model1']
    results_df['error_model2'] = results_df[target] - results_df[target+'_model2']
    for i in range(1, 3):
        metrics['model_'+str(i)] = {
            'mae': mean_absolute_error(results_df[target], results_df[f'{target}_model{i}']),
            'mse': mean_squared_error(results_df[target], results_df[f'{target}_model{i}']),
            'rmse': mean_squared_error(results_df[target], results_df[f'{target}_model{i}'])**0.5,
            'r2_': r2_score(results_df[target], results_df[f'{target}_model{i}'])
    }
    metrics_df = pd.DataFrame(metrics)
    results_df.to_csv(save_dir / f'{data.stem}_results.csv')
    metrics_df.to_csv(save_dir / f'{data.stem}_metrics.csv')

def metrics_vanillalstm():
    df = pd.read_csv('data/errors_vanillalstm.csv')
    models = [model.split('error_')[1] for model in df.columns if 'error_' in model]
    metrics = {}
    for model in models:
        metrics[model] = {
            'mae': mean_absolute_error(df['RUL'], df[f'RUL_{model}']),
            'mse': mean_squared_error(df['RUL'], df[f'RUL_{model}']),
            'rmse': mean_squared_error(df['RUL'], df[f'RUL_{model}'])**0.5,
            'r2': r2_score(df['RUL'], df[f'RUL_{model}'])
        }
    metrics_df = pd.DataFrame(metrics).transpose()
    metrics_df.to_csv('results/errors_vanillalstm_metrics.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train datasets using XGBoost and DecisionTreeRegressor')
    parser.add_argument('-d', '--data', type=str, help='Path to the dataset', required=True)
    parser.add_argument('--sep', type=str, help='Separator for the dataset', default=',')
    parser.add_argument('-t', '--target', type=str, help='Target column', required=True)
    parser.add_argument('--save', type=str, help='Path to save the results', default='results')
    parser.add_argument('--categorical', nargs='+', help='Categorical features to be one-hot encoded')
    parser.add_argument('--drop', nargs='+', help='Columns to be dropped')
    args = parser.parse_args()
    train_dataset(
        data=Path(args.data),
        sep=args.sep,
        save_dir=Path(args.save),
        target=args.target,
        categorical_features=args.categorical,
        to_drop=args.drop)