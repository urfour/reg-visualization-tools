import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from utils.models import CMAPSSTraining
from utils.losses.pytorch import mse, mae, quad_quad
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

def train_all():
    """ Train models on all datasets and save results and metrics to csv files """
    train_metrics()
    train_under_over()
    train_different_errors()
    train_apartments()
    train_cmapss()

def plot_all():
    """ Generate all plots """
    df_biases = pd.read_csv('results/metrics_biases.csv')
    df_under_over = pd.read_csv('results/under_over.csv')
    df_diff_errors = pd.read_csv('results/different_errors.csv')
    df_cmapss = pd.read_csv('results/errors_cmapss.csv')
    df_apartments = pd.read_csv('results/apartments_results.csv')
    path = 'all_fig'
    models_cmapss = ('se', 'quad_quad_0.01')
    other_models = ('model1', 'model2')

    # Generated datasets
    plot_distributions_alone(data=df_biases, path=path, models=other_models, file_name='fig1.png', model_index=1)
    plot_distributions(data=df_under_over, path=path, models=other_models, file_name='fig2.png')
    plot_diff_distributions(data=df_diff_errors, path=path, models=other_models, file_name='fig3.png')
    # Real datasets
    plot_predicted_real(data=df_cmapss, target_name='RUL', path=path, models=models_cmapss, file_name='fig4.png')
    plot_distributions_alone(data=df_cmapss, path=path, models=models_cmapss, file_name='fig5.png')
    plot_predicted_real_multiple(data=df_cmapss, target_name='RUL', path=path, models=models_cmapss, file_name='fig6.png')
    plot_errors(data=df_cmapss, path=path, models=models_cmapss, show_one_individual=True, index=[47, 800], file_name='fig7.png')
    plot_errors(data=df_cmapss, path=path, models=models_cmapss, file_name='fig8.png')
    plot_hourglass(data=df_cmapss, path=path, models=models_cmapss, file_name='fig9.png')
    plot_mean_median(data=df_cmapss, path=path, models=models_cmapss, file_name='fig10.png')
    plot_distributions(data=df_cmapss, path=path, models=models_cmapss, file_name='fig11.png')
    plot_density_proximity(data=df_cmapss, path=path, models=models_cmapss, file_name='fig12.png')
    plot_errors(data=df_apartments, path=path, models=other_models, file_name='fig13.png')
    plot_with_proximity(data=df_apartments, path=path, models=other_models, file_name='fig14.png', with_hourglass=False)
    plot_compared_proximity(data=df_cmapss, path=path, models=models_cmapss, file_name='fig15.png')

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