from typing import Union
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    results_df[target+'_model1'] = pred_1
    results_df[target+'_model2'] = pred_2
    results_df['error_model1'] = results_df[target] - results_df[target+'_model1']
    results_df['error_model2'] = results_df[target] - results_df[target+'_model2']

    metrics = {
        'mae_model1': mean_absolute_error(results_df[target], results_df[target+'_model1']),
        'mae_model2': mean_absolute_error(results_df[target], results_df[target+'_model2']),
        'mse_model1': mean_squared_error(results_df[target], results_df[target+'_model1']),
        'mse_model2': mean_squared_error(results_df[target], results_df[target+'_model2']),
        'rmse_model1': mean_squared_error(results_df[target], results_df[target+'_model1'])**0.5,
        'rmse_model2': mean_squared_error(results_df[target], results_df[target+'_model2'])**0.5
    }
    metrics_df = pd.DataFrame(metrics, index=[0])
    
    results_df.to_csv(save_dir / f'{data.stem}_results.csv')
    metrics_df.to_csv(save_dir / f'{data.stem}_metrics.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train datasets using XGBoost and DecisionTreeRegressor')
    parser.add_argument('--data', type=str, help='Path to the dataset', required=True)
    parser.add_argument('--sep', type=str, help='Separator for the dataset', default=',')
    parser.add_argument('--target', type=str, help='Target column', required=True)
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