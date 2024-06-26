import itertools
import pandas as pd
from typing import Union
from os import makedirs
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from os.path import join
from scipy.spatial.distance import mahalanobis

plt.rcParams.update({'font.size': 20})
plt.rcParams['xtick.major.pad'] = '8'
plt.rcParams['ytick.major.pad'] = '8'

def plot_predicted_real(data : pd.DataFrame, target_name : str, path : str, models : Union[tuple, str] = 'all', with_outliers = False):
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
        to_save = path
    for combination in all_metrics_combination:
        if models == 'all':
            to_save = join(path, combination[0]+'_'+combination[1])
        makedirs(to_save, exist_ok=True)
        extrema = data[f'{target_name}_{combination[0]}'].max()
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
        fig.savefig(join(to_save, 'actual_predicted.png'))
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
        to_save = path
    for combination in all_metrics_combination:
        if models == 'all':
            to_save = join(path, combination[0]+'_'+combination[1])
        makedirs(to_save, exist_ok=True)
        extrema = abs(data[[f'{target_name}_{model}' for model in combination]].max().max())
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_xlim(0, extrema)
        ax.set_ylim(0, extrema)
        ax.set_aspect('equal', adjustable='box')

        x = data[target_name]
        y = data[f'{target_name}_{combination[0]}']
        y2 = data[f'{target_name}_{combination[1]}']
        ax.scatter(x, y, c='tab:orange', s=50, label='Model 1')
        ax.scatter(x, y2, c='tab:green', s=50, label="Model 2")

        ax.set_xlabel('Predicted values')
        ax.set_ylabel('Real values')

        # Diagonals
        ax.plot([0, extrema], [0, extrema], color='tab:blue', linewidth=2)

        fig.tight_layout()
        fig.legend()
        fig.savefig(join(to_save, 'actual_predicted_two.png'))
        plt.close()

def plot_errors(data : pd.DataFrame, path : str, models : Union[tuple, str] = 'all', show_one_individual = True):
    """ Plot only the errors of the models.
    
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    show_one_individual (bool, optional): If True, show the coordinates of one individual. Defaults to True.
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
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
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

        if show_one_individual:
            point = data[['error_'+model for model in combination]].sort_values(by='error_'+combination[0]).iloc[2].to_numpy()
            ax.plot([-extrema, point[0]], [point[1], point[1]], color='tab:gray', linestyle='--')
            ax.plot([point[0], point[0]], [-extrema, point[1]], color='tab:gray', linestyle='--')
            ax.text(point[0]+2, point[1], f'({int(point[0])}, {int(point[1])})', ha='left', va='center', color='tab:gray', fontsize=15)
        
        fig.tight_layout()
        fig.savefig(join(to_save, 'errors.png'))
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
        to_save = path
    for combination in all_metrics_combination:
        if models == 'all':
            to_save = join(path, combination[0]+'_'+combination[1])
        makedirs(to_save, exist_ok=True)
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.set_xlim(-extrema, extrema)
        ax.set_ylim(-extrema, extrema)
        ax.set_aspect('equal', adjustable='box')
        # Vertical axis
        ax.plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        # Horizontal axis
        ax.plot([-extrema, extrema], [0, 0], color='black', linewidth=1)
        # Diagonal
        equal_points, = ax.plot([-extrema, extrema], [-extrema, extrema], linewidth=1, label="Equal points")
        ax.plot([-extrema, extrema], [extrema, -extrema], color='tab:blue', linewidth=1)

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

        fig.legend(handles=[equal_points], loc='lower right')
        fig.tight_layout()
        fig.savefig(join(to_save, 'density.png'))
        plt.close()

def plot_errors_vs_density(data : pd.DataFrame, path : str, models : Union[tuple, str] = 'all'):
    """ Plot the figure with the errors and the density of the points for the models.

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
        to_save = path
    for combination in all_metrics_combination:
        if models == 'all':
            to_save = join(path, combination[0]+'_'+combination[1])
        makedirs(to_save, exist_ok=True)
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
        fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

        ax[0].set_xlim(-extrema, extrema)
        ax[0].set_ylim(-extrema, extrema)
        ax[0].set_aspect('equal', adjustable='box')

        ax[0].plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        ax[0].plot([-extrema, extrema], [0, 0], color='black', linewidth=1)

        ax[0].plot([-extrema, extrema], [-extrema, extrema], color='tab:blue', linewidth=1)
        ax[0].plot([-extrema, extrema], [extrema, -extrema], color='tab:blue', linewidth=1)

        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]
        ax[0].scatter(x, y, c='black', s=100)

        ax[0].set_xlabel('Errors of model 1')
        ax[0].set_ylabel('Errors of model 2')

        ax[1].set_aspect('equal', adjustable='box')
        ax[1].plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        ax[1].plot([-extrema, extrema], [0, 0], color='black', linewidth=1)
        equal_points,  = ax[1].plot([-extrema, extrema], [-extrema, extrema], linewidth=1, label="Equal points")
        ax[1].plot([-extrema, extrema], [extrema, -extrema], color='tab:blue', linewidth=1)
        
        median = (data['error_'+combination[0]].median(), data['error_'+combination[1]].median())
        distance = np.sqrt((x - median[0])**2 + (y - median[1])**2)
        data['distance'] = distance

        data = data.sort_values(by='distance')
        data['percentile'] = data['distance'].apply(lambda x: (len(data[data['distance'] <= x]) / len(data)) * 100)
        data = data.sort_index()

        density = ax[1].scatter(x, y, c=data['percentile'], s=100, cmap='Spectral')
        fig.colorbar(density, label="Percentile", fraction=0.028, orientation='horizontal')
        ax[1].plot(median[0], median[1], 'x', color='black', markersize=5)
        ax[1].set_xlabel('Errors of model 1')

        fig.legend(handles=[equal_points], loc='upper right')
        fig.tight_layout()
        fig.savefig(join(to_save, 'errors_vs_density.png'))
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
        to_save = path
    for combination in all_metrics_combination:
        if models == 'all':
            to_save = join(path, combination[0]+'_'+combination[1])
        makedirs(to_save, exist_ok=True)
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
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
        fig.savefig(join(to_save, 'mean.png'))
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
        to_save = path
    for combination in all_metrics_combination:
        if models == 'all':
            to_save = join(path, combination[0]+'_'+combination[1])
        makedirs(to_save, exist_ok=True)
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
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
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        density = ax.scatter(x, y, c=z, s=100)
        fig.colorbar(density, label="KDE", fraction=0.030)
        
        ax.set_xlabel('Errors of model 1')
        ax.set_ylabel('Errors of model 2')

        fig.legend(handles=[mean_line, std_line], loc='lower right', bbox_to_anchor=(0.84, 0.14))
        # fig.tight_layout()
        fig.savefig(join(to_save, 'mean_density.png'))
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
        to_save = path
    for combination in all_metrics_combination:
        if models == 'all':
            to_save = join(path, combination[0]+'_'+combination[1])
        makedirs(to_save, exist_ok=True)
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
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
        fig.savefig(join(to_save, 'hourglass.png'))
        plt.close()

def plot_distributions_alone(data : pd.DataFrame, path : str, models : Union[tuple, str] = 'all'):
    """ Plot the figure with the distribution of errors for each model alone

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
        to_save = path
    for combination in all_metrics_combination:
        if models == 'all':
            to_save = join(path, combination[0]+'_'+combination[1])
        makedirs(to_save, exist_ok=True)
        for model in combination:
            extrema = max(abs(data['error_'+model].min()), abs(data['error_'+model].max()))
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharex=True, sharey=True)

            hist = ax.hist(data['error_'+model], bins=50, alpha=0.8, edgecolor='black', color='tab:orange')
            ax.grid(True, linestyle='--', alpha=0.5)
            ticks = ax.get_yticks()
            ticks = [tick for tick in ticks if tick != 0]
            ax.set_yticks(ticks)

            ax.set_xlim(-extrema, extrema)
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Errors')

            max_bin_height = max(hist[0])
            max_bin_index = np.where(hist[0] == max_bin_height)[0][0]
            mode = (hist[1][max_bin_index] + hist[1][max_bin_index + 1]) / 2

            ax.plot([mode, mode], [0, max_bin_height], color='tab:blue', linestyle='--', label=f'Mode')
            ax.text(mode, max_bin_height, f'{mode:.2f}', ha='center', va='bottom', color='tab:blue', fontsize=15)

            ax.legend()
            fig.tight_layout()
            fig.savefig(join(to_save, f'distributions_{model}.png'))

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
        to_save = path
    for combination in all_metrics_combination:
        if models == 'all':
            to_save = join(path, combination[0]+'_'+combination[1])
        makedirs(to_save, exist_ok=True)
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
        fig, ax = plt.subplots(2, 1, figsize=(16, 16), sharex=True, sharey=True)
        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]

        ax[0].hist(x, bins=50, alpha=0.5, label='Model 1', edgecolor='black', color='tab:orange')
        ax[0].grid(True, linestyle='--', alpha=0.5)
        ax[0].set_xlim(-extrema, extrema)

        ax[1].hist(y, bins=50, alpha=0.5, label='Model 2', edgecolor='black', color='tab:green')
        ax[1].grid(True, linestyle='--', alpha=0.5)
        ax[1].set_xlim(-extrema, extrema)

        ax[0].set_ylabel('Frequency')
        ax[1].set_xlabel('Errors')
        ax[1].set_ylabel('Frequency')
        fig.legend(loc='upper right')
        fig.subplots_adjust(hspace=0)
        fig.tight_layout()
        fig.savefig(join(to_save, 'distributions.png'))

def plot_with_proximity(
        data : pd.DataFrame, path : str, models : Union[tuple, str] = 'all', 
        colormap : str = 'Spectral', file_name : str = 'circle_plot.png',
        distance_metric : str = 'euclidean', with_hourglass : bool = False):
    """ Plot the figure with the proximity of the points.

    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    colormap (str, optional): The colormap to use. Defaults to 'Spectral'.
    file_name (str, optional): The name of the file to save. Defaults to 'circle_plot.png'.
    distance_metric (str, optional): The distance to use. Defaults to 'euclidean'.
    with_hourglass (bool, optional): If True, plot the hourglass. Defaults to False.
    """
    data = data.copy()
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
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
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

        if with_hourglass:
            abs_better, _ = ax.fill(
                [-extrema, 0, extrema], [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema],  
                c='tab:orange', 
                alpha=0.2, 
                label=f"Model 1 is better")
            ord_better, _ = ax.fill(
                [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema], [-extrema, 0, extrema], 
                c='tab:green', 
                alpha=0.2, 
                label=f"Model 2 is better")

        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]

        # Calculate distance to the median
        median = (data['error_'+combination[0]].median(), data['error_'+combination[1]].median())
        if distance_metric == 'manhattan':
            distance = np.abs(x - median[0]) + np.abs(y - median[1])
        elif distance_metric == 'mahalanobis':
            cov = np.cov(data[['error_'+combination[0], 'error_'+combination[1]]], rowvar=False)
            # calculate coordinates for each point
            distance = []
            for row in data[['error_'+combination[0], 'error_'+combination[1]]].values:
                distance.append(mahalanobis(row, median, np.linalg.inv(cov)))
        else:
            distance = np.sqrt((x - median[0])**2 + (y - median[1])**2)
        data['distance'] = distance

        data = data.sort_values(by='distance')
        # Get for each point, the percentage of points that are at least that distance
        data['percentile'] = data['distance'].apply(lambda x: (len(data[data['distance'] <= x]) / len(data)) * 100)

        # Reorder the points
        data = data.sort_index()

        density = ax.scatter(x, y, c=data['percentile'], s=100, cmap=colormap)
        fig.colorbar(density, label="Percentile", fraction=0.030)

        # draw a cross on the median point (median[0], median[1])
        ax.plot(median[0], median[1], 'x', color='black', markersize=5)

        ax.set_xlabel(f'Errors of model 1', fontsize=20, labelpad=10)
        ax.set_ylabel(f'Errors of model 2', fontsize=20, labelpad=10)
        if with_hourglass:
            ax.xaxis.label.set_color('tab:orange')
            ax.yaxis.label.set_color('tab:green')
            fig.legend(handles=[abs_better, ord_better], loc='lower right')
        else:
            fig.legend(handles=[equal_points], loc='lower right')
        fig.subplots_adjust(left=0.15)
        fig.savefig(join(to_save, file_name))
        plt.close()

def plot_everything(data : pd.DataFrame, path : str, models : Union[tuple, str] = 'all', show_one_individual = False, 
                    distance_metric : str = 'euclidean', colormap : str = 'Spectral'):
    """ Plot all the figures for the models.
        
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    show_one_individual (bool, optional): If True, shows the dash lines for one individual. Defaults to False.
    distance_metric (str, optional): The distance to use. Defaults to 'euclidean'.
    colormap (str, optional): The colormap to use. Defaults to 'Spectral'.
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
        extrema = max(abs(data[['error_'+model for model in combination]].min().min()), abs(data[['error_'+model for model in combination]].max().max()))
        makedirs(to_save, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(24, 24))
        ax.set_xlim(-extrema, extrema)
        ax.set_ylim(-extrema, extrema)
        ax.set_aspect('equal', adjustable='box')
        # Vertical axis
        ax.plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        # Horizontal axis
        ax.plot([-extrema, extrema], [0, 0], color='black', linewidth=1)
        # Diagonals
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
        # xy = np.vstack([x, y])
        # # Order the points by density
        # z = gaussian_kde(xy)(xy)
        # idx = z.argsort()
        # x, y, z = x[idx], y[idx], z[idx]
        # density = ax.scatter(x, y, c=z, s=100)
        # fig.colorbar(density, label="KDE", fraction=0.030)

        # Calculate distance to the mean
        mean = (data['error_'+combination[0]].mean(), data['error_'+combination[1]].mean())
        if distance_metric == 'manhattan':
            distance = np.abs(x - mean[0]) + np.abs(y - mean[1])
        elif distance_metric == 'mahalanobis':
            cov = np.cov(data[['error_'+combination[0], 'error_'+combination[1]]], rowvar=False)
            distance = data[['error_'+combination[0], 'error_'+combination[1]]].apply(lambda x: mahalanobis(x, mean, cov), axis=1)
        else:
            distance = np.sqrt((x - mean[0])**2 + (y - mean[1])**2)
        data['distance'] = distance

        data = data.sort_values(by='distance')
        # Get for each point, the percentage of points that are at least that distance
        data['percentile'] = data['distance'].apply(lambda x: (len(data[data['distance'] <= x]) / len(data)) * 100)

        # Reorder the points
        data = data.sort_index()

        density = ax.scatter(x, y, c=data['percentile'], s=100, cmap=colormap)
        fig.colorbar(density, label="Percentile", fraction=0.030)

        mean = (data['error_'+combination[0]].mean(), data['error_'+combination[1]].mean())
        std = (np.sqrt(data['error_'+combination[0]].std()), np.sqrt(data['error_'+combination[1]].std()))

        mean_line, = ax.plot([mean[0], mean[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Mean')
        std_line, = ax.plot([mean[0] - 2*std[0], mean[0] - 2*std[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Std Deviation', alpha=0.5)
        ax.plot([mean[0] + 2*std[0], mean[0] + 2*std[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Std Deviation', alpha=0.5)

        ax.plot([-extrema, extrema], [mean[1], mean[1]], color='tab:blue', linestyle='--')
        ax.plot([-extrema, extrema], [mean[1] - 2*std[1], mean[1] - 2*std[1]], color='tab:blue', linestyle='--', alpha=0.5)
        ax.plot([-extrema, extrema], [mean[1] + 2*std[1], mean[1] + 2*std[1]], color='tab:blue', linestyle='--', alpha=0.5)

        ax.set_xlabel('Errors of model 1', fontsize=20, labelpad=10)
        ax.xaxis.label.set_color('tab:orange')
        ax.set_ylabel('Errors of model 2', fontsize=20, labelpad=10)
        ax.yaxis.label.set_color('tab:green')

        # Dash-lines to show one individual
        if show_one_individual:
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

def draw_all_plots(data, path, models, target_name = None):
    """ Draw all the plots for the models.

    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    target_name (str): The name of the target column.
    path (str): The path to save the generated plot(s).
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    """
    if target_name is not None:
        plot_predicted_real(data=data, target_name=target_name, path=path, models=models)
        plot_predicted_real_multiple(data=data, target_name=target_name, path=path, models=models)
    plot_errors(data=data, path=path, models=models)
    plot_density(data=data, path=path, models=models)
    plot_hourglass(data=data, path=path, models=models)
    plot_mean(data=data, path=path, models=models)
    plot_mean_density(data=data, path=path, models=models)
    plot_distributions_alone(data=data, path=path, models=models)
    plot_distributions(data=data, path=path, models=models)
    plot_everything(data=data, path=path, models=models)
    plot_with_proximity(data=data, path=path, models=models)

df = pd.read_csv('data/errors_vanillalstm.csv')
selected_models = ('se', 'quad_quad_0.01')
target_name = 'RUL'
path = 'fig/vanillalstm'
plot_density(df, path, models=selected_models)
plot_with_proximity(df, path, models=selected_models, colormap='Spectral', distance_metric='euclidean', file_name='circle_plot', with_hourglass=False)
# plot_with_proximity(df, path, models=selected_models, colormap='Spectral', distance_metric='euclidean', file_name='circle_plot_euclidean.png', with_hourglass=False)
# plot_with_proximity(df, path, models=selected_models, colormap='Spectral', distance_metric='mahalanobis', file_name='circle_plot_mahalanobis.png', with_hourglass=False)

# plot_with_proximity(df, path, models=selected_models, colormap='Spectral', distance_metric='euclidean', file_name='circle_plot_euclidean.png')
# plot_with_proximity(df, path, models=selected_models, colormap='Spectral', distance_metric='mahalanobis', file_name='circle_plot_mahalanobis.png')
# plot_with_proximity(df, path, models=selected_models, colormap='Spectral', distance_metric='manhattan', file_name='circle_plot_manhattan.png')
# plot_distributions_alone(df, path, models=selected_models)
# draw_all_plots(data=df, target_name=target_name, path=path, models=selected_models)

# df_abalone = pd.read_csv('results/abalone_results.csv')
# df_bike = pd.read_csv('results/bike_results.csv')
# df_wine = pd.read_csv('results/wine_results.csv')
# # draw_all_plots(data=df_abalone, path='fig/abalone', models=selected_models)
# # draw_all_plots(data=df_bike, path='fig/bike', models=selected_models)
# # draw_all_plots(data=df_wine, path='fig/wine', models=selected_models)

# df_apartments = pd.read_csv('results/apartments_results.csv')
# selected_models = 'all'
# plot_errors_vs_density(df_apartments, 'fig/apartments', models=selected_models)