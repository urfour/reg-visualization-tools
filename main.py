import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from utils.models import CMAPSSTraining, LSTMTraining
from utils.losses.pytorch import mse, mae, lin_se, lin_lin, QuadQuad
from utils.visualization import *
import numpy as np
from argparse import ArgumentParser
from scipy.stats import gaussian_kde
np.random.seed(0)


def _resolve_plot_languages(lang_arg: str):
    if lang_arg == 'both':
        return ['en', 'fr']
    return [lang_arg]

### Generated datasets
def train_metrics():
    """ Generate data to illustrate metrics biases
    and save results and metrics to csv files"""
    truth = np.arange(1, 1001)
    pred1 = np.append(np.arange(1, 1000), 1500)
    pred2 = np.arange(1, 1001) + np.random.normal(0, 10, 1000)
    df = pd.DataFrame({
        'values': truth,
        'values_model1': pred1,
        'error_model1': pred1 - truth,
        'values_model2': pred2,
        'error_model2': pred2 - truth
        })
    metrics = {
        'rmse_model1': np.sqrt(mean_squared_error(truth, pred1)),
        'rmse_model2': np.sqrt(mean_squared_error(truth, pred2)),
        'mae_model1': mean_absolute_error(truth, pred1),
        'mae_model2': mean_absolute_error(truth, pred2)
    }
    df.to_csv('results/metrics_biases.csv')
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv('results/metrics_biases_metrics.csv', index=False)

def train_under_over():
    """ Generate data to simulate under estimations and over estimations
    and save results and metrics to csv files"""
    # Dataset 1 (under-estimations)
    n = 1000
    truth = np.arange(1, n + 1)
    pred1 = np.concatenate([
        truth[:750] - np.random.uniform(1, 20, 750),
        truth[750:] + np.random.uniform(-10, 10, 250)
    ])
    # Dataset 2 (over-estimations)
    pred2 = np.concatenate([
        truth[:750] + np.random.uniform(1, 20, 750),
        truth[750:] + np.random.uniform(-10, 10, 250)
    ])
    df = pd.DataFrame({
        'values': truth,
        'values_model1': pred1,
        'values_model2': pred2,
        'error_model1': pred1 - truth,
        'error_model2': pred2 - truth
    })
    metrics = {
        'rmse_model1': np.sqrt(mean_squared_error(truth, pred1)),
        'rmse_model2': np.sqrt(mean_squared_error(truth, pred2)),
        'mae_model1': mean_absolute_error(truth, pred1),
        'mae_model2': mean_absolute_error(truth, pred2)
    }
    df.to_csv('results/under_over.csv')
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv('results/under_over_metrics.csv', index=False)

def train_different_errors():
    """ Generate data to simulate different errors """
    n_samples = 1000
    truth = np.random.rand(n_samples) * 1000

    errors_model_1 = np.linspace(-10, 10, n_samples) * np.sin(np.linspace(0, 2 * np.pi, n_samples))
    errors_model_2 = np.linspace(10, -10, n_samples) * np.cos(np.linspace(0, 2 * np.pi, n_samples))

    pred1 = truth + errors_model_1
    pred2 = truth + errors_model_2

    df = pd.DataFrame({
        'values': truth,
        'values_model1': pred1,
        'error_model1': pred1 - truth,
        'values_model2': pred2,
        'error_model2': pred2 - truth
        })
    metrics = {
        'rmse_model1': np.sqrt(mean_squared_error(truth, pred1)),
        'rmse_model2': np.sqrt(mean_squared_error(truth, pred2)),
        'mae_model1': mean_absolute_error(truth, pred1),
        'mae_model2': mean_absolute_error(truth, pred2)
    }
    df.to_csv('results/different_errors.csv')
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv('results/different_errors_metrics.csv', index=False)

### Real datasets
def train_apartments():
    """ Train models on apartments data and save results and metrics to csv files """
    df = pd.read_csv('data/apartments.csv', sep=';', low_memory=False)

    # Data cleaning
    df = df[['bathrooms', 'bedrooms', 'square_feet', 'cityname', 'price']]
    df = df.dropna()
    df['cityname'] = df['cityname'].apply(lambda x: x.strip())
    location_stats = df.groupby('cityname')['cityname'].agg('count').sort_values(ascending=False)
    location_stats_10 = location_stats[location_stats <= 10]
    df.cityname = df.cityname.apply(lambda x: 'other' if x in location_stats_10 else x)

    # Train-test split
    cityname = pd.get_dummies(df['cityname'], dummy_na=True)
    X = pd.concat([df.drop(['cityname', 'price'], axis=1), cityname], axis=1)
    X.columns = X.columns.astype(str)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

    # Model training
    results_df = pd.DataFrame()
    metrics_df = pd.DataFrame()

    model1 = DecisionTreeRegressor()
    model1.fit(X_train, y_train)
    model2 = LinearRegression()
    model2.fit(X_train, y_train)

    pred_1 = model1.predict(X_test)
    pred_2 = model2.predict(X_test)

    results_df['price'] = y_test
    results_df['price_model1'] = pred_1
    results_df['price_model2'] = pred_2
    results_df['error_model1'] = results_df['price'] - results_df['price_model1']
    results_df['error_model2'] = results_df['price'] - results_df['price_model2']

    metrics = {
        'mae_model1': mean_absolute_error(results_df['price'], results_df['price_model1']),
        'mae_model2': mean_absolute_error(results_df['price'], results_df['price_model2']),
        'mse_model1': mean_squared_error(results_df['price'], results_df['price_model1']),
        'mse_model2': mean_squared_error(results_df['price'], results_df['price_model2']),
        'rmse_model1': mean_squared_error(results_df['price'], results_df['price_model1'])**0.5,
        'rmse_model2': mean_squared_error(results_df['price'], results_df['price_model2'])**0.5
    }
    metrics_df = pd.DataFrame(metrics, index=[0])

    results_df.to_csv('results/apartments_results.csv')
    metrics_df.to_csv('results/apartments_metrics.csv')

def train_cmapss():
    """ Train models on CMAPSS data and save results and metrics to csv files """
    all_losses = {
        'A1': mse,
        'A2': mae,
        'A3': lin_se(0.3),
        'A4': lin_se(0.4),
        'A5': lin_se(0.5),
        'A6': lin_se(0.6),
        'A7': lin_lin(0.01, 1.0),
        'A8': lin_lin(0.05, 1.0),
        'A9': lin_lin(0.2, 1.0),
        'A10': lin_lin(0.3, 1.0),
        'A11': QuadQuad(0.01),
        'A12': QuadQuad(0.03)
    }
    df = pd.DataFrame()
    for loss in all_losses:
        print('Loss: {}'.format(loss))
        training = CMAPSSTraining(dataset='FD001',
                                model_type='vanillalstm',
                                criterion=all_losses[loss])
        training.train(save_model=False, verbose=False)
        df = pd.concat([df, training.calc_errors(loss)], axis=1)
    df.T.drop_duplicates().T.to_csv('results/cmapss_errors.csv')
    
    all_models = [col.split('error_')[1] for col in df.columns if 'error_' in col]
    rmse_values = {}
    mae_values = {}
    for model in all_models:
        rmse_values[model] = round(np.sqrt(np.mean(df[f'error_{model}']**2)), 2)
        mae_values[model] = round(np.mean(np.abs(df[f'error_{model}'])), 2)
    metrics_df = pd.DataFrame({'rmse': rmse_values, 'mae': mae_values})
    metrics_df.to_csv('results/cmapss_metrics.csv')

def train_ai4i():
    df = pd.read_csv('data/ai4i2020.csv')
    df_processed = df.drop(columns=['UDI', 'Product ID'])
    df_processed = pd.get_dummies(df_processed, columns=['Type'], prefix='Type')
    bool_columns = df_processed.select_dtypes(include=['bool']).columns
    df_processed[bool_columns] = df_processed[bool_columns].astype(int)
    
    failure_indices = df_processed[df_processed['Machine failure'] == 1].index.tolist()
    df_processed['RUL'] = np.nan
    
    for idx in df_processed.index:
        next_failures = [f_idx for f_idx in failure_indices if f_idx >= idx]
        if next_failures:
            next_failure = next_failures[0]
            df_processed.loc[idx, 'RUL'] = next_failure - idx
        else:
            df_processed.loc[idx, 'RUL'] = df_processed.index[-1] - idx
    
    df_features = df_processed.copy()
    df_features['temp_ratio'] = df_features['Process temperature [K]'] / df_features['Air temperature [K]']
    df_features['temp_diff'] = df_features['Process temperature [K]'] - df_features['Air temperature [K]']
    df_features['temp_product'] = df_features['Process temperature [K]'] * df_features['Air temperature [K]']
    df_features['power_ratio'] = df_features['Rotational speed [rpm]'] / (df_features['Torque [Nm]'] + 1e-6)
    df_features['power_product'] = df_features['Rotational speed [rpm]'] * df_features['Torque [Nm]']
    df_features['specific_power'] = df_features['power_product'] / (df_features['Process temperature [K]'] + 1e-6)
    df_features['wear_per_rpm'] = df_features['Tool wear [min]'] / (df_features['Rotational speed [rpm]'] + 1e-6)
    df_features['wear_per_torque'] = df_features['Tool wear [min]'] / (df_features['Torque [Nm]'] + 1e-6)
    df_features['wear_temp_interaction'] = df_features['Tool wear [min]'] * df_features['temp_diff']
    df_features['efficiency'] = df_features['Rotational speed [rpm]'] / (df_features['power_product'] + 1e-6)
    df_features['thermal_stress'] = df_features['temp_diff'] * df_features['power_product']
    df_features['load_factor'] = df_features['Torque [Nm]'] / df_features['Rotational speed [rpm]'] * 1000
    
    target = 'RUL'
    feature_cols = [col for col in df_features.columns if col not in ['RUL', 'Machine failure']]
    X = df_features[feature_cols]
    y = df_features[target]
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    selector = SelectKBest(score_func=f_regression, k=min(50, X_scaled.shape[1]//2))
    X_selected = selector.fit_transform(X_scaled, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    results = {}
    print("Training models on AI4I dataset:")
    print(f"\n LSTM 1")
    criterion_1 = QuadQuad(0.2)
    lstm_1 = LSTMTraining(X_train, y_train.values, X_test, criterion=criterion_1)
    lstm_1.train()
    y_pred_lstm = lstm_1.predict()
    
    mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
    mse_lstm = mean_squared_error(y_test, y_pred_lstm)
    r2_lstm = r2_score(y_test, y_pred_lstm)
    
    results['LSTM_1'] = {
        'mae': mae_lstm,
        'mse': mse_lstm,
        'r2': r2_lstm,
        'cv_mae': mae_lstm,
        'predictions': y_pred_lstm
    }
    
    print(f"    MAE: {mae_lstm:.3f}")
    print(f"    RMSE: {np.sqrt(mse_lstm):.3f}")
    print(f"    R²: {r2_lstm:.3f}")

    print("LSTM 2")
    criterion_2 = QuadQuad(0.8)
    lstm_2 = LSTMTraining(X_train, y_train.values, X_test, criterion=criterion_2)
    lstm_2.train()
    y_pred_lstm = lstm_2.predict()

    # Métriques LSTM
    mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
    mse_lstm = mean_squared_error(y_test, y_pred_lstm)
    r2_lstm = r2_score(y_test, y_pred_lstm)
    
    results['LSTM_2'] = {
        'mae': mae_lstm,
        'mse': mse_lstm,
        'r2': r2_lstm,
        'cv_mae': mae_lstm,
        'predictions': y_pred_lstm
    }
    
    print(f"    MAE: {mae_lstm:.3f}")
    print(f"    RMSE: {np.sqrt(mse_lstm):.3f}")
    print(f"    R²: {r2_lstm:.3f}")

    results_data = {'RUL_true': y_test}
    
    for name in results.keys():
        predictions = results[name]['predictions']
        errors = predictions - y_test.values
        
        results_data[f'RUL_pred_{name}'] = predictions
        results_data[f'error_{name}'] = errors
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('results/ai4i2020_results.csv', index=False)
    
    metrics_df = pd.DataFrame(results).T
    # remove predictions
    metrics_df = metrics_df.drop(columns=['predictions'])
    metrics_df.to_csv('results/ai4i2020_metrics.csv')
        
    return results

def train_all():
    """ Train models on all datasets and save results and metrics to csv files """
    print("Training all datasets...")
    train_metrics()
    train_under_over()
    train_different_errors()
    train_apartments()
    train_cmapss()
    train_ai4i()

def plot_kde(data : pd.DataFrame, path : str, file_name = 'distribution.pdf', 
                             models : Union[tuple, str] = 'all', model_index = 0):
    """ Plot the figure with the probability density of errors for each model alone using KDE

    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'distribution.pdf'.
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    model_index (int, optional): The index of the model to plot (0 or 1). Defaults to 0.
    """

    if models == 'all':
        all_metrics = [col.split('error_')[1] for col in data.columns if 'error_' in col]
        all_metrics_combination = list(itertools.combinations(all_metrics, 2))
    else:
        all_metrics_combination = [models]
        to_save = path
    for combination in all_metrics_combination:
        if models == 'all':
            to_save = join(path, combination[0]+'_'+combination[1])
        makedirs(to_save, exist_ok=True)
        extrema = max(abs(data['error_'+combination[model_index]].min()), abs(data['error_'+combination[model_index]].max()))
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharex=True, sharey=True)
        if model_index == 0:
            color = 'tab:orange'
        else:
            color = 'tab:green'
        
        # Calculate KDE
        errors = data['error_'+combination[model_index]]
        kde = gaussian_kde(errors)
        x = np.linspace(-extrema, extrema, 1000)
        y = kde(x)

        # Plot density
        ax.plot(x, y, color=color, label='Density')
        ax.fill_between(x, y, color=color, alpha=0.3)

        ax.grid(True, linestyle='--', alpha=0.5)
        ticks = ax.get_yticks()
        ticks = [tick for tick in ticks if tick != 0]
        ax.set_yticks(ticks)

        ax.set_xlim(-extrema, extrema)
        ax.set_ylabel('Probability Density')
        ax.set_xlabel('Errors')

        median = errors.median()
        median_height = kde(median)[0]
        
        ax.axvline(0, color='black')
        ax.plot([median, median], [0, median_height], color='tab:blue', linestyle='--', label=f'Median')
        ax.text(median, median_height, f'{median:.2f}', ha='center', va='bottom', color='tab:blue', fontsize=15)

        ax.legend()
        fig.tight_layout()
        fig.savefig(join(to_save, file_name))

def plot_histogram(data : pd.DataFrame, path : str, file_name = 'histogram.pdf', 
                             models : Union[tuple, str] = 'all', model_index = 0):
    """ Plot the figure with the histogram of errors for each model alone
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'histogram.pdf'.
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    model_index (int, optional): The index of the model to plot (0 or 1). Defaults to 0.
    """
    if models == 'all':
        all_metrics = [col.split('error_')[1] for col in data.columns if 'error_' in col]
        all_metrics_combination = list(itertools.combinations(all_metrics, 2))
    else:
        all_metrics_combination = [models]
        to_save = path
    for combination in all_metrics_combination:
        if models == 'all':
            to_save = join(path, combination[0]+'_'+combination[1])
        makedirs(to_save, exist_ok=True)
        extrema = max(abs(data['error_'+combination[0]].min()), abs(data['error_'+combination[0]].max()),
                       abs(data['error_'+combination[1]].min()), abs(data['error_'+combination[1]].max()))
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if model_index == 0:
            color = 'tab:orange'
        else:
            color = 'tab:green'
        ax.hist(data['error_'+combination[model_index]], bins=30, color=color, alpha=0.7, density=False)
        ax.set_xlim(-extrema, extrema)
        ax.set_ylabel('Counts')
        ax.set_xlabel('Errors')
        ax.grid(True, linestyle='--', alpha=0.5)
        ticks = ax.get_yticks()
        ticks = [tick for tick in ticks if tick != 0]
        ax.set_yticks(ticks)

        median = data['error_'+combination[model_index]].median()
        ax.axvline(0, color='black')
        ax.axvline(median, color='tab:blue', linestyle='--', label=f'Median')
        ax.text(median, ax.get_ylim()[1]*0.9, f'{median:.2f}', ha='center', va='bottom', color='tab:blue', fontsize=15)
        ax.legend()
        fig.tight_layout()
        fig.savefig(join(to_save, file_name))


def plot_all(language: str = 'en', output_path: str = 'all_fig'):
    """ Generate all plots """
    df_biases = pd.read_csv('results/metrics_biases.csv')
    df_under_over = pd.read_csv('results/under_over.csv')
    df_diff_errors = pd.read_csv('results/different_errors.csv')
    df_cmapss = pd.read_csv('results/cmapss_errors.csv')
    df_apartments = pd.read_csv('results/apartments_results.csv')
    df_ai4i = pd.read_csv('results/ai4i2020_results.csv')
    df_seoul = pd.read_csv('results/seoul_results.csv')
    path = output_path
    models_cmapss = ('D1', 'D10')
    other_models = ('model1', 'model2')

    # plot_histogram(data=df_cmapss, path=path, models=models_cmapss, file_name='cmapss_histogram.pdf', model_index=0)
    # Generated datasets
    plot_distributions(data=df_biases, path=path, models=other_models, file_name='fig1.pdf', labels=('A1', 'A2'), share_axes=False, use_log_scale=True, lang=language)
    plot_distributions(data=df_under_over, path=path, models=other_models, file_name='fig2.pdf', labels=('B1', 'B2'), lang=language)
    plot_diff_distributions(data=df_diff_errors, path=path, models=other_models, file_name='fig3.pdf', lang=language)
    # Real datasets
    plot_predicted_real(data=df_cmapss, target_name='RUL', path=path, models=models_cmapss, file_name='fig4.pdf', lang=language)
    plot_predicted_real_multiple(data=df_cmapss, target_name='RUL', path=path, models=models_cmapss, file_name='fig5.pdf', labels=('D1', 'D10'), lang=language)
    plot_errors_boxplot(data=df_cmapss, path=path, file_name='fig6.pdf', lang=language)
    plot_predicted_real_grid(data=df_cmapss, target_name='RUL', path=path, file_name='fig7.pdf', lang=language)
    plot_hourglass(data=df_cmapss, path=path, models=models_cmapss, file_name='fig8.pdf', labels=('D1', 'D10'), lang=language)
    # plot_mean_median(data=df_cmapss, path=path, models=models_cmapss, file_name='fig8a.pdf', with_hourglass=False)
    # plot_distributions(data=df_cmapss, path=path, models=models_cmapss, file_name='fig8b.pdf')
    plot_density(data=df_cmapss, path=path, models=models_cmapss, file_name='fig9.pdf', labels=('D1', 'D10'), lang=language)
    plot_hexbins(data=df_cmapss, path=path, models=models_cmapss, file_name='fig10.pdf', labels=('D1', 'D10'), lang=language)
    plot_with_proximity(data=df_cmapss, path=path, models=models_cmapss, file_name='fig11.pdf', distance_metric='euclidean', labels=('D1', 'D10'), lang=language)
    plot_with_proximity(data=df_apartments, path=path, models=other_models, file_name='apartments.pdf', distance_metric='euclidean', labels=('E1', 'E2'), lang=language)
    plot_with_proximity(data=df_cmapss, path=path, models=models_cmapss, file_name='fig12.pdf', distance_metric='mahalanobis', labels=('D1', 'D10'), lang=language)
    plot_with_proximity(data=df_ai4i, path=path, models=('LSTM_1', 'LSTM_2'), file_name='fig13.pdf', labels=('F1', 'F2'), lang=language)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--train-dataset', type=str, help='Train the choosen dataset(s). Available: metrics, underover, differrors, apartments, cmapss, all')
    parser.add_argument('-a', '--train-all', action='store_true', help='Train all datasets')
    parser.add_argument('-p', '--plot', action='store_true', help='Generate all plots')
    parser.add_argument('--lang', choices=['en', 'fr', 'both'], default='en', help='Language for plot labels')
    args = parser.parse_args()
    if args.train_dataset is not None:
        if args.train_dataset == 'all':
            train_all()
        elif args.train_dataset == 'metrics':
            train_metrics()
        elif args.train_dataset == 'underover':
            train_under_over()
        elif args.train_dataset == 'differrors':
            train_different_errors()
        elif args.train_dataset == 'apartments':
            train_apartments()
        elif args.train_dataset == 'cmapss':
            train_cmapss()
        elif args.train_dataset == 'ai4i':
            train_ai4i()
        else:
            print('Invalid dataset name')
            sys.exit(1)
    elif args.train_all:
        train_all()
    if args.plot:
        for lang in _resolve_plot_languages(args.lang):
            output_path = 'all_fig' if args.lang != 'both' else join('all_fig', lang)
            plot_all(language=lang, output_path=output_path)
    else:
        if args.train_dataset is None and not args.train_all:
            parser.print_help()