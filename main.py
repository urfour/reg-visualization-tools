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
from utils.losses.pytorch import mse, mae, quad_quad, QuadQuad
from utils.visualization import *
import numpy as np
from argparse import ArgumentParser
np.random.seed(0)

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
        'se': mse,
        'ae': mae,
        'quad_quad_0.01': quad_quad(0.01)
    }
    df = pd.DataFrame()
    for loss in all_losses:
        print('Loss: {}'.format(loss))
        training = CMAPSSTraining(dataset='FD001',
                                model_type='vanillalstm',
                                criterion=all_losses[loss])
        training.train(save_model=False, verbose=False)
        df = pd.concat([df, training.calc_errors(loss)], axis=1)
    df.T.drop_duplicates().T.to_csv('results/errors_cmapss.csv')

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
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }
    
    results = {}
    print("Training models on AI4I dataset:")
    
    for name, model in models.items():
        print(f"📈 {name}...")
        
        # Entraînement
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Métriques
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        results[name] = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'cv_mae': cv_mae,
            'predictions': y_pred
        }
        
        print(f"    MAE: {mae:.3f} (CV: {cv_mae:.3f})")
        print(f"    R²: {r2:.3f}")
    
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
        'cv_mae': mae_lstm
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
        'cv_mae': mae_lstm
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
    metrics_df.to_csv('results/ai4i2020_metrics.csv')
        
    return results

def train_all():
    """ Train models on all datasets and save results and metrics to csv files """
    train_metrics()
    train_under_over()
    train_different_errors()
    train_apartments()
    train_cmapss()
    train_ai4i()

def plot_all():
    """ Generate all plots """
    df_biases = pd.read_csv('results/metrics_biases.csv')
    df_under_over = pd.read_csv('results/under_over.csv')
    df_diff_errors = pd.read_csv('results/different_errors.csv')
    df_cmapss = pd.read_csv('results/errors_cmapss.csv')
    df_cmapss_vanilla = pd.read_csv('results/errors_vanillalstm.csv')
    df_cmapss_metrics = pd.read_csv('results/errors_vanillalstm_metrics.csv', index_col=0)
    df_apartments = pd.read_csv('results/apartments_results.csv')
    df_ai4i = pd.read_csv('results/improved_rul_predictions.csv')
    path = 'all_fig'
    models_cmapss = ('se', 'quad_quad_0.01')
    other_models = ('model1', 'model2')

    # Generated datasets
    plot_distributions_alone(data=df_biases, path=path, models=other_models, file_name='fig1.pdf', model_index=1)
    plot_distributions(data=df_under_over, path=path, models=other_models, file_name='fig2.pdf', labels=('15', '16'))
    plot_diff_distributions(data=df_diff_errors, path=path, models=other_models, file_name='fig3.pdf')
    # Real datasets
    plot_predicted_real(data=df_cmapss, target_name='RUL', path=path, models=models_cmapss, file_name='fig4.pdf')
    # plot_distributions_alone(data=df_cmapss, path=path, models=models_cmapss, file_name='fig5.pdf')
    plot_predicted_real_multiple(data=df_cmapss, target_name='RUL', path=path, models=models_cmapss, file_name='fig5.pdf', labels=('1', '11'))
    plot_errors_boxplot(data=df_cmapss_vanilla, metrics=df_cmapss_metrics, path=path, file_name='fig7.pdf')
    plot_predicted_real_grid(data=df_cmapss_vanilla, metrics=df_cmapss_metrics, target_name='RUL', path=path, file_name='fig8.pdf')
    # plot_errors(data=df_cmapss, path=path, models=models_cmapss, show_one_individual=True, index=[47, 800], file_name='fig7.pdf')
    # plot_errors(data=df_cmapss, path=path, models=models_cmapss, file_name='fig8.pdf')
    plot_hourglass(data=df_cmapss, path=path, models=models_cmapss, file_name='fig9.pdf', labels=('1', '11'))
    # plot_mean_median(data=df_cmapss, path=path, models=models_cmapss, file_name='fig10.pdf', with_hourglass=False)
    # plot_distributions(data=df_cmapss, path=path, models=models_cmapss, file_name='fig11.pdf')
    plot_density(data=df_cmapss, path=path, models=models_cmapss, file_name='fig10.pdf', labels=('1', '11'))
    plot_hexbins(data=df_cmapss, path=path, models=models_cmapss, file_name='fig11.pdf', labels=('1', '11'))
    plot_with_proximity(data=df_cmapss, path=path, models=models_cmapss, file_name='fig12.pdf', distance_metric='euclidean', labels=('1', '11'))
    plot_with_proximity(data=df_apartments, path=path, models=other_models, file_name='fig13.pdf', distance_metric='euclidean', labels=('19', '20'))
    plot_with_proximity(data=df_cmapss, path=path, models=models_cmapss, file_name='fig14.pdf', distance_metric='mahalanobis', labels=('1', '11'))
    plot_with_proximity(data=df_ai4i, path=path, models=('LSTM_1', 'LSTM_2'), file_name='fig15.pdf', labels=('21', '22'))

    # Generated datasets
    # plot_distributions_alone(data=df_ai4i, path=path, models=other_models, file_name='new_fig1.png', model_index=1)
    # plot_distributions(data=df_ai4i, path=path, models=other_models, file_name='new_fig2.png')
    # # Real datasets
    # plot_predicted_real(data=df_ai4i, target_name='Rotational speed [rpm]', path=path, models=other_models, file_name='new_fig4.png')
    # plot_distributions_alone(data=df_ai4i, path=path, models=other_models, file_name='new_fig5.png')
    # plot_predicted_real_multiple(data=df_ai4i, target_name='Rotational speed [rpm]', path=path, models=other_models, file_name='new_fig6.png')
    # plot_errors(data=df_ai4i, path=path, models=other_models, file_name='new_fig8.png')
    # plot_hourglass(data=df_ai4i, path=path, models=other_models, file_name='new_fig9.png')
    # plot_mean_median(data=df_ai4i, path=path, models=other_models, file_name='new_fig10.png', with_hourglass=False)
    # plot_distributions(data=df_ai4i, path=path, models=other_models, file_name='new_fig11.png')
    # plot_density_proximity(data=df_ai4i, path=path, models=other_models, file_name='new_fig12.png')
    # plot_errors_vs_density(data=df_ai4i, path=path, models=other_models, file_name='new_fig13.png')
    # plot_compared_proximity(data=df_ai4i, path=path, models=other_models, file_name='new_fig14.png')
    # plot_hexbins(data=df_ai4i, path=path, models=other_models, file_name='new_fig15.png')
    # plot_with_proximity(data=df_ai4i, path=path, models='all', file_name='new_fig16.png')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--train-dataset', type=str, help='Train the choosen dataset(s). Available: metrics, underover, differrors, apartments, cmapss, all')
    parser.add_argument('-a', '--train-all', action='store_true', help='Train all datasets')
    parser.add_argument('-p', '--plot', action='store_true', help='Generate all plots')
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
        plot_all()
    else:
        if args.train_dataset is None and not args.train_all:
            parser.print_help()