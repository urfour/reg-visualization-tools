import itertools
import pandas as pd
from typing import Union
from os import makedirs
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, wilcoxon
from os.path import join

plt.rcParams.update({'font.size': 20})
plt.rcParams['xtick.major.pad'] = '8'
plt.rcParams['ytick.major.pad'] = '8'

def plot_predicted_real(data : pd.DataFrame, target_name : str, path : str, models : Union[tuple, str] = 'all'):
    """ 
    Plot actual values vs predicted values of the models.

    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    target_name (str): The name of the target variable.
    path (str): The path to save the generated plot(s).
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    """
    if models == 'all':
        all_metrics = [col.split('error_')[1] for col in data.columns if 'error_' in col]
        all_metrics_combination = list(itertools.combinations(all_metrics, 2))
    else:
        all_metrics_combination = [models]
    for combination in all_metrics_combination:
        extrema = data[f'{target_name}_{combination[0]}'].max()
        makedirs(path, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_xlim(0, extrema)
        ax.set_ylim(0, extrema)
        ax.set_aspect('equal', adjustable='box')

        x = data[target_name]
        y = data[f'{target_name}_{combination[0]}']
        ax.scatter(x, y, c='black', s=50)

        ax.set_xlabel('Real value')
        ax.set_ylabel('Predicted value')

        # Diagonals
        ax.plot([0, extrema], [0, extrema], color='tab:blue', linewidth=2)

        fig.tight_layout()
        fig.savefig(join(path, 'actual_predicted.png'))
        plt.close()

def plot_predicted_real_multiple(data : pd.DataFrame, target_name : str, path : str, models : Union[tuple, str] = 'all'):
    """ Plot actual values vs predicted values for two models.
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    target_name (str): The name of the target variable.
    path (str): The path to save the generated plot(s).
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    """
    if models == 'all':
        all_metrics = [col.split('error_')[1] for col in data.columns if 'error_' in col]
        all_metrics_combination = list(itertools.combinations(all_metrics, 2))
    else:
        all_metrics_combination = [models]
    for combination in all_metrics_combination:
        extrema = abs(data[[f'{target_name}_{model}' for model in combination]].max().max())
        makedirs(path, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_xlim(0, extrema)
        ax.set_ylim(0, extrema)
        ax.set_aspect('equal', adjustable='box')

        x = data[target_name]
        y = data[f'{target_name}_{combination[0]}']
        y2 = data[f'{target_name}_{combination[1]}']
        ax.scatter(x, y, c='tab:orange', s=50)
        ax.scatter(x, y2, c='tab:green', s=50)

        ax.set_xlabel('Model 1')
        ax.xaxis.label.set_color('tab:orange')
        ax.set_ylabel('Model 2')
        ax.yaxis.label.set_color('tab:green')

        # Diagonals
        ax.plot([0, extrema], [0, extrema], color='tab:blue', linewidth=2)

        fig.tight_layout()
        fig.savefig(join(path, 'actual_predicted_two.png'))
        plt.close()

def plot_errors(data : pd.DataFrame, path : str, models : Union[tuple, str] = 'all'):
    """ Plot only the errors of the models.
    
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    """
    if models == 'all':
        all_metrics = [col.split('error_')[1] for col in data.columns if 'error_' in col]
        all_metrics_combination = list(itertools.combinations(all_metrics, 2))
    else:
        all_metrics_combination = [models]
    for combination in all_metrics_combination:
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
        makedirs(path, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        ax.set_xlim(-extrema, extrema)
        ax.set_ylim(-extrema, extrema)
        ax.set_aspect('equal', adjustable='box')
        # Vertical axis
        ax.plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        # Horizontal axis
        ax.plot([-extrema, extrema], [0, 0], color='black', linewidth=1)

        # Diagonals
        ax.plot([-extrema, extrema], [-extrema, extrema], color='tab:blue', linewidth=1)
        ax.plot([-extrema, extrema], [extrema, -extrema], color='tab:blue', linewidth=1)

        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]
        ax.scatter(x, y, c='black', s=100)

        ax.set_xlabel('Errors of model 1')
        ax.set_ylabel('Errors of model 2')

        # Dash-lines to show one individual
        point = data[['error_'+model for model in combination]].sort_values(by='error_'+combination[0]).iloc[2].to_numpy()
        ax.plot([-extrema, point[0]], [point[1], point[1]], color='tab:gray', linestyle='--')
        ax.plot([point[0], point[0]], [-extrema, point[1]], color='tab:gray', linestyle='--')

        # Add coordinates of the point
        ax.text(point[0]+2, point[1], f'({int(point[0])}, {int(point[1])})', ha='left', va='center', color='tab:gray', fontsize=15)
        fig.tight_layout()
        fig.savefig(join(path, f'errors.png'))
        plt.close()    

def plot_density(data : pd.DataFrame, path : str, models : Union[tuple, str] = 'all'):
    """ Plot the figure only with the points ordered by density for the models.
        
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    """
    if models == 'all':
        all_metrics = [col.split('error_')[1] for col in data.columns if 'error_' in col]
        all_metrics_combination = list(itertools.combinations(all_metrics, 2))
    else:
        all_metrics_combination = [models]
    for combination in all_metrics_combination:
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
        makedirs(path, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.set_xlim(-extrema, extrema)
        ax.set_ylim(-extrema, extrema)
        ax.set_aspect('equal', adjustable='box')
        # Vertical axis
        ax.plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        # Horizontal axis
        ax.plot([-extrema, extrema], [0, 0], color='black', linewidth=1)

        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]
        xy = np.vstack([x, y])
        # Order the points by density
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        #density2 = ax.scatter(x, y, c=z, s=100, alpha=0.1)
        density = ax.scatter(x, y, c=z, s=100)
        fig.colorbar(density, label="KDE", fraction=0.030)

        ax.set_xlabel('Errors of model 1')
        ax.set_ylabel('Errors of model 2')

        fig.tight_layout()
        fig.savefig(join(path, f'density.png'))
        plt.close()

def plot_mean(data : pd.DataFrame, path : str, models : Union[tuple, str] = 'all'):
    """ Plot the figure with the mean dash lines for the models.
        
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    """
    if models == 'all':
        all_metrics = [col.split('error_')[1] for col in data.columns if 'error_' in col]
        all_metrics_combination = list(itertools.combinations(all_metrics, 2))
    else:
        all_metrics_combination = [models]
    for combination in all_metrics_combination:
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
        makedirs(path, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        ax.set_xlim(-extrema, extrema)
        ax.set_ylim(-extrema, extrema)
        ax.set_aspect('equal', adjustable='box')
        # Vertical axis
        ax.plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        # Horizontal axis
        ax.plot([-extrema, extrema], [0, 0], color='black', linewidth=1)
        
        mean = (data['error_'+combination[0]].mean(), data['error_'+combination[1]].mean())
        std = (np.sqrt(data['error_'+combination[0]].std()), np.sqrt(data['error_'+combination[1]].std()))

        mean_line, = ax.plot([mean[0], mean[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Mean')
        std_line, = ax.plot([mean[0] - std[0], mean[0] - std[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Std Deviation', alpha=0.5)
        ax.plot([mean[0] + std[0], mean[0] + std[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Std Deviation', alpha=0.5)

        ax.plot([-extrema, extrema], [mean[1], mean[1]], color='tab:blue', linestyle='--')
        ax.plot([-extrema, extrema], [mean[1] - std[1], mean[1] - std[1]], color='tab:blue', linestyle='--', alpha=0.5)
        ax.plot([-extrema, extrema], [mean[1] + std[1], mean[1] + std[1]], color='tab:blue', linestyle='--', alpha=0.5)

        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]

        ax.scatter(x, y, s=100, c='black', alpha=0.9)

        ax.set_xlabel('Errors of model 1')
        ax.set_ylabel('Errors of model 2')

        fig.legend(handles=[mean_line, std_line], loc='lower right', bbox_to_anchor=(0.97, 0.09))
        fig.tight_layout()
        fig.savefig(join(path, f'mean.png'))
        plt.close()

def plot_mean_density(data : pd.DataFrame, path : str, models : Union[tuple, str] = 'all'):
    """ Plot the figure with the mean dash lines for the models and the density.
        
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    """
    if models == 'all':
        all_metrics = [col.split('error_')[1] for col in data.columns if 'error_' in col]
        all_metrics_combination = list(itertools.combinations(all_metrics, 2))
    else:
        all_metrics_combination = [models]
    for combination in all_metrics_combination:
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
        makedirs(path, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        ax.set_xlim(-extrema, extrema)
        ax.set_ylim(-extrema, extrema)
        ax.set_aspect('equal', adjustable='box')
        # Vertical axis
        ax.plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        # Horizontal axis
        ax.plot([-extrema, extrema], [0, 0], color='black', linewidth=1)
        
        mean = (data['error_'+combination[0]].mean(), data['error_'+combination[1]].mean())
        std = (np.sqrt(data['error_'+combination[0]].std()), np.sqrt(data['error_'+combination[1]].std()))

        mean_line, = ax.plot([mean[0], mean[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Mean')
        std_line, = ax.plot([mean[0] - std[0], mean[0] - std[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Std Deviation', alpha=0.5)
        ax.plot([mean[0] + std[0], mean[0] + std[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Std Deviation', alpha=0.5)

        ax.plot([-extrema, extrema], [mean[1], mean[1]], color='tab:blue', linestyle='--')
        ax.plot([-extrema, extrema], [mean[1] - std[1], mean[1] - std[1]], color='tab:blue', linestyle='--', alpha=0.5)
        ax.plot([-extrema, extrema], [mean[1] + std[1], mean[1] + std[1]], color='tab:blue', linestyle='--', alpha=0.5)

        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]
        xy = np.vstack([x, y])
        # Order the points by density
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        density = ax.scatter(x, y, c=z, s=100)
        fig.colorbar(density, label="KDE", fraction=0.030)
        
        ax.set_xlabel('Errors of model 1')
        ax.set_ylabel('Errors of model 2')

        fig.legend(handles=[mean_line, std_line], loc='lower right', bbox_to_anchor=(0.84, 0.14))
        # fig.tight_layout()
        fig.savefig(join(path, f'mean_density.png'))
        plt.close()

def plot_hourglass(data : pd.DataFrame, path : str, models : Union[tuple, str] = 'all'):
    """ Plot the figure only with the hourglass for the models
        
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    """
    if models == 'all':
        all_metrics = [col.split('error_')[1] for col in data.columns if 'error_' in col]
        all_metrics_combination = list(itertools.combinations(all_metrics, 2))
    else:
        all_metrics_combination = [models]
    for combination in all_metrics_combination:
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
        makedirs(path, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        ax.set_xlim(-extrema, extrema)
        ax.set_ylim(-extrema, extrema)
        ax.set_aspect('equal', adjustable='box')
        # Vertical axis
        ax.plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        # Horizontal axis
        ax.plot([-extrema, extrema], [0, 0], color='black', linewidth=1)
        # Diagonal
        equal_points, = ax.plot([-extrema, extrema], [-extrema, extrema], label="Equal points")
        ax.plot([-extrema, extrema], [extrema, -extrema], color='tab:blue', linewidth=1)

        # Model in abs is better
        abs_better, _ = ax.fill(
            [-extrema, 0, extrema], [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema],  
            c='tab:orange', 
            alpha=0.2, 
            label=f"Model 1 is better")
        # Model in ord is better
        ord_better, _ = ax.fill(
            [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema], [-extrema, 0, extrema], 
            c='tab:green', 
            alpha=0.2, 
            label=f"Model 2 is better")
        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]

        ax.scatter(x, y, s=100, c='black', alpha=0.9)

        ax.set_xlabel('Errors of model 1')
        ax.xaxis.label.set_color('tab:orange')
        ax.set_ylabel('Errors of model 2')
        ax.yaxis.label.set_color('tab:green')

        fig.tight_layout()
        fig.legend(handles=[abs_better, ord_better, equal_points], loc='lower right', bbox_to_anchor=(0.97, 0.07))
        fig.savefig(join(path, f'hourglass.png'))
        plt.close()

def plot_distributions(data : pd.DataFrame, path : str, models : Union[tuple, str] = 'all'):
    """ Plot the figure with the distributions of the errors for the models
        
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    """
    if models == 'all':
        all_metrics = [col.split('error_')[1] for col in data.columns if 'error_' in col]
        all_metrics_combination = list(itertools.combinations(all_metrics, 2))
    else:
        all_metrics_combination = [models]
    for combination in all_metrics_combination:
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
        makedirs(path, exist_ok=True)
        fig, ax = plt.subplots(2, 1, figsize=(16, 16), sharex=True, sharey=True)
        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]

        freq_x, _ = np.histogram(x, bins=50)
        freq_y, _ = np.histogram(y, bins=50)

        max_freq = max(freq_x.max(), freq_y.max())

        ax[0].hist(x, bins=50, alpha=0.5, label='Model 1', edgecolor='black', color='tab:orange')
        ax[0].grid(True, linestyle='--', alpha=0.5)
        ax[0].set_xlim(-extrema, extrema)
        # ax[0].set_ylim(0, max_freq)

        ax[1].hist(y, bins=50, alpha=0.5, label='Model 2', edgecolor='black', color='tab:green')
        ax[1].grid(True, linestyle='--', alpha=0.5)
        ax[1].set_xlim(-extrema, extrema)
        # ax[1].set_ylim(0, max_freq)

        # ax[0].set_xlabel('Errors')
        ax[0].set_ylabel('Frequency')
        
        ax[1].set_xlabel('Errors')
        ax[1].set_ylabel('Frequency')
        fig.legend(loc='upper right')
        fig.subplots_adjust(hspace=0)
        fig.tight_layout()
        fig.savefig(join(path, f'distributions.png'))

def plot_everything(data : pd.DataFrame, path : str, models : Union[tuple, str] = 'all'):
    """ Plot all the figures for the models.
        
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    """
    if models == 'all':
        all_metrics = [col.split('error_')[1] for col in data.columns if 'error_' in col]
        all_metrics_combination = list(itertools.combinations(all_metrics, 2))
    else:
        all_metrics_combination = [models]
    for combination in all_metrics_combination:
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
        to_save = join(path, combination[0]+'_'+combination[1])
        makedirs(to_save, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(24, 24))
        ax.set_xlim(-extrema, extrema)
        ax.set_ylim(-extrema, extrema)
        ax.set_aspect('equal', adjustable='box')
        # Vertical axis
        ax.plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        # Horizontal axis
        ax.plot([-extrema, extrema], [0, 0], color='black', linewidth=1)
        # Diagonal
        equal_points, = ax.plot([-extrema, extrema], [-extrema, extrema], label="Equal points")

        # Model in abs is better
        abs_better, _ = ax.fill(
            [-extrema, 0, extrema], [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema],  
            c='tab:orange', 
            alpha=0.2, 
            label=f"Model 1 is better")
        # Model in ord is better
        ord_better, _ = ax.fill(
            [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema], [-extrema, 0, extrema], 
            c='tab:green', 
            alpha=0.2, 
            label=f"Model 2 is better")
        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]
        xy = np.vstack([x, y])
        # Order the points by density
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        #density2 = ax.scatter(x, y, c=z, s=100, alpha=0.1)
        density = ax.scatter(x, y, c=z, s=100)

        mean = (data['error_'+combination[0]].mean(), data['error_'+combination[1]].mean())
        std = (np.sqrt(data['error_'+combination[0]].std()), np.sqrt(data['error_'+combination[1]].std()))

        mean_line, = ax.plot([mean[0], mean[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Mean')
        std_line, = ax.plot([mean[0] - std[0], mean[0] - std[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Std Deviation', alpha=0.5)
        ax.plot([mean[0] + std[0], mean[0] + std[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Std Deviation', alpha=0.5)

        ax.plot([-extrema, extrema], [mean[1], mean[1]], color='tab:blue', linestyle='--')
        ax.plot([-extrema, extrema], [mean[1] - std[1], mean[1] - std[1]], color='tab:blue', linestyle='--', alpha=0.5)
        ax.plot([-extrema, extrema], [mean[1] + std[1], mean[1] + std[1]], color='tab:blue', linestyle='--', alpha=0.5)

        fig.colorbar(density, label="KDE", fraction=0.030)
        ax.set_xlabel('Errors of model 1', fontsize=20, labelpad=10)
        ax.xaxis.label.set_color('tab:orange')
        ax.set_ylabel('Errors of model 2', fontsize=20, labelpad=10)
        ax.yaxis.label.set_color('tab:green')

        # Dash-lines to show one individual
        point = data[['error_'+model for model in combination]].sort_values(by='error_'+combination[0]).iloc[3].to_numpy()
        ax.plot([-extrema, point[0]], [point[1], point[1]], color='black', linestyle='--')
        ax.plot([point[0], point[0]], [-extrema, point[1]], color='black', linestyle='--')

        # Add histogram
        pos = ax.get_position()
        ax_histx = fig.add_axes([pos.x0+0.025, 0.15, pos.width-0.02, 0.08])
        ax_histy = fig.add_axes([0.15, pos.y0+0.015, 0.08, pos.height-0.025])

        # Remove axis
        ax_histx.spines['top'].set_visible(False)
        ax_histx.spines['right'].set_visible(False)
        ax_histx.spines['left'].set_visible(False)
        ax_histx.spines['bottom'].set_visible(False)
        ax_histx.yaxis.set_visible(False)
        ax_histx.xaxis.set_visible(False)

        ax_histy.spines['top'].set_visible(False)
        ax_histy.spines['right'].set_visible(False)
        ax_histy.spines['left'].set_visible(False)
        ax_histy.spines['bottom'].set_visible(False)
        ax_histy.yaxis.set_visible(False)
        ax_histy.xaxis.set_visible(False)

        # Remove background
        ax_histx.set_facecolor('none')
        ax_histy.set_facecolor('none')

        ax_histx.hist(x, bins=50, color='tab:orange', edgecolor='black', alpha=0.5)
        ax_histy.hist(y, bins=50, orientation='horizontal', color='tab:green', edgecolor='black', alpha=0.5)
        ax_histx.grid(True, linestyle='--', alpha=0.5)
        ax_histy.grid(True, linestyle='--', alpha=0.5)

        ax_histx.set_xlim(ax.get_xlim())
        ax_histy.set_ylim(ax.get_ylim())

        ax_histx.set_xticklabels([])
        ax_histy.set_yticklabels([])

        fig.legend(handles=[abs_better, ord_better, equal_points, mean_line, std_line], loc='lower right')
        fig.subplots_adjust(left=0.15)
        fig.savefig(join(to_save, 'general_plot.png'))
        plt.close()

df_vanillalstm = pd.read_csv('errors/errors_se.csv')
selected_models = 'all'

# plot_predicted_real(data=df_vanillalstm, target_name='RUL', path='explanation_figures', models=selected_models)
# plot_predicted_real_multiple(data=df_vanillalstm, target_name='RUL', path='explanation_figures', models=selected_models)
# plot_errors(data=df_vanillalstm, path='explanation_figures', models=selected_models)
# plot_density(data=df_vanillalstm, path='explanation_figures', models=selected_models)
# plot_hourglass(data=df_vanillalstm, path='explanation_figures', models=selected_models)
# plot_mean(data=df_vanillalstm, path='explanation_figures', models=selected_models)
# plot_mean_density(data=df_vanillalstm, path='explanation_figures', models=selected_models)
# plot_distributions(data=df_vanillalstm, path='explanation_figures', models=selected_models)
plot_everything(data=df_vanillalstm, path='errors/fig', models=selected_models)