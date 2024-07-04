import itertools
import pandas as pd
from typing import Union
from os import makedirs
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from os.path import join
from scipy.spatial.distance import mahalanobis

plt.rcParams.update({'font.size': 25})
plt.rcParams['xtick.major.pad'] = '8'
plt.rcParams['ytick.major.pad'] = '8'

def plot_predicted_real(data : pd.DataFrame, target_name : str, path : str, 
                        file_name = 'actual_predicted.png', models : Union[tuple, str] = 'all'):
    """ 
    Plot actual values vs predicted values of the models.

    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    target_name (str): The name of the target variable.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'actual_predicted.png'.
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

        # Diagonals
        equal_points, = ax.plot([0, extrema], [0, extrema], color='tab:blue', linewidth=2, label='Equal errors')

        x = data[target_name]
        y = data[f'{target_name}_{combination[0]}']
        ax.scatter(x, y, c='black', s=50)

        ax.set_xlabel('Real value')
        ax.set_ylabel('Predicted value')

        # Diagonals
        ax.plot([0, extrema], [0, extrema], color='tab:blue', linewidth=2)

        fig.tight_layout()
        fig.legend(handles=[equal_points], loc='lower right', bbox_to_anchor=(0.97, 0.12))
        fig.savefig(join(to_save, file_name))
        plt.close()

def plot_predicted_real_multiple(data : pd.DataFrame, target_name : str, path : str, 
                                 file_name = 'actual_predicted_two.png', models : Union[tuple, str] = 'all'):
    """ Plot actual values vs predicted values for two models.
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    target_name (str): The name of the target variable.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'actual_predicted_two.png'.
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
        ax.scatter(x, y, c='tab:orange', s=50, label=f'Model 1')
        ax.scatter(x, y2, c='tab:green', s=50, label=f'Model 2')

        ax.set_xlabel('Predicted values')
        ax.set_ylabel('Real values')

        # Diagonals
        ax.plot([0, extrema], [0, extrema], color='tab:blue', linewidth=2)

        fig.tight_layout()
        fig.legend(loc='lower right', bbox_to_anchor=(0.97, 0.12))
        fig.savefig(join(to_save, file_name))
        plt.close()

def plot_errors(data : pd.DataFrame, path : str, file_name = 'errors.png',
                models : Union[tuple, str] = 'all', show_one_individual = False, index = [0]):
    """ Plot only the errors of the models.
    
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'errors.png'.
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    show_one_individual (bool, optional): If True, show the coordinates of one individual. Defaults to True.
    index (list, optional): The list of the index of individuals to show. Defaults to [0].
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
        equal_points, = ax.plot([-extrema, extrema], [-extrema, extrema], color='tab:blue', linewidth=1, label='Equal errors')
        ax.plot([-extrema, extrema], [extrema, -extrema], color='tab:blue', linewidth=1)

        if show_one_individual:
            for i, num_index in enumerate(index):
                point = data[['error_'+model for model in combination]].iloc[num_index].to_numpy()
                ax.scatter(point[0], point[1], c='black', s=300)
                ax.plot([-extrema, point[0]], [point[1], point[1]], color='tab:gray', linestyle='--', linewidth=2)
                ax.plot([point[0], point[0]], [-extrema, point[1]], color='tab:gray', linestyle='--', linewidth=2)
                ax.text(point[0]+2, point[1], f'{chr(65+i)} ({int(point[0])}, {int(point[1])})', ha='left', va='center', color='tab:gray')
        else:
            x = data['error_'+combination[0]]
            y = data['error_'+combination[1]]
            ax.scatter(x, y, c='black', s=100)

        ax.set_xlabel(f'Errors of model 1')
        ax.set_ylabel(f'Errors of model 2')
        
        fig.tight_layout()
        fig.legend(handles=[equal_points], loc='lower right')
        fig.savefig(join(to_save, file_name))
        plt.close()    

def plot_density(data : pd.DataFrame, path : str, file_name = 'density.png', models : Union[tuple, str] = 'all'):
    """ Plot the figure only with the points ordered by density for the models.
        
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'density.png'.
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
        equal_points, = ax.plot([-extrema, extrema], [-extrema, extrema], linewidth=1, label="Equal errors")
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

        ax.set_xlabel(f'Errors of model 1')
        ax.set_ylabel(f'Errors of model 2')

        fig.legend(handles=[equal_points], loc='lower right')
        fig.tight_layout()
        fig.savefig(join(to_save, file_name))
        plt.close()

def plot_errors_vs_density(data : pd.DataFrame, path : str, 
                           file_name = 'errors_vs_density.png', models : Union[tuple, str] = 'all'):
    """ Plot the figure with the errors and the density of the points for the models.

    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'errors_vs_density.png'.
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

        ax[0].set_xlabel(f'Errors of model 1')
        ax[0].set_ylabel(f'Errors of model 2')

        ax[1].set_aspect('equal', adjustable='box')
        ax[1].plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        ax[1].plot([-extrema, extrema], [0, 0], color='black', linewidth=1)
        equal_points, = ax[1].plot([-extrema, extrema], [-extrema, extrema], linewidth=1, label="Equal errors")
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
        ax[1].set_xlabel(f'Errors of model 1')

        fig.legend(handles=[equal_points], loc='upper right')
        fig.tight_layout()
        fig.savefig(join(to_save, file_name))
        plt.close()

def plot_mean(data : pd.DataFrame, path : str, file_name = 'mean.png', models : Union[tuple, str] = 'all'):
    """ Plot the figure with the mean dash lines for the models.
        
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'mean.png'.
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

        ax.set_xlabel(f'Errors of model 1')
        ax.set_ylabel(f'Errors of model 2')

        fig.legend(handles=[mean_line, std_line], loc='lower right', bbox_to_anchor=(0.97, 0.09))
        fig.tight_layout()
        fig.savefig(join(to_save, file_name))
        plt.close()

def plot_mean_median(data : pd.DataFrame, path : str, file_name = 'mean_median.png', models : Union[tuple, str] = 'all'):
    """ Plot the figure with the mean dash lines for the models and the median.

    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'mean_median.png'.
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
        fig, ax = plt.subplots(1, 2, figsize=(16, 16), sharex=True, sharey=True)
        ax[0].set_xlim(-extrema, extrema)
        ax[0].set_ylim(-extrema, extrema)
        ax[0].set_aspect('equal', adjustable='box')
        # Vertical axis
        ax[0].plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        # Horizontal axis
        ax[0].plot([-extrema, extrema], [0, 0], color='black', linewidth=1)
        
        mean = (data['error_'+combination[0]].mean(), data['error_'+combination[1]].mean())
        std = (np.sqrt(data['error_'+combination[0]].std()), np.sqrt(data['error_'+combination[1]].std()))
        median = (data['error_'+combination[0]].median(), data['error_'+combination[1]].median())

        median_line, = ax[0].plot([median[0], median[0]], [-extrema, extrema], color='tab:red', linestyle='--', label='Median', linewidth=2)
        ax[0].plot([-extrema, extrema], [median[1], median[1]], color='tab:red', linestyle='--', linewidth=2)
        mean_line, = ax[0].plot([mean[0], mean[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Mean', linewidth=2)
        ax[0].plot([-extrema, extrema], [mean[1], mean[1]], color='tab:blue', linestyle='--', linewidth=2)

        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]

        ax[0].scatter(x, y, s=100, c='black')

        ax[0].set_xlabel(f'Errors of model 1')
        ax[0].set_ylabel(f'Errors of model 2')

        ax[1].set_xlim(-extrema, extrema)
        ax[1].set_ylim(-extrema, extrema)
        ax[1].set_aspect('equal', adjustable='box')
        # Vertical axis
        ax[1].plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        # Horizontal axis
        ax[1].plot([-extrema, extrema], [0, 0], color='black', linewidth=1)

        ax[1].plot([mean[0], mean[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Mean', linewidth=2)
        std_line, = ax[1].plot([mean[0] - std[0], mean[0] - std[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Std Deviation', alpha=0.5, linewidth=2)
        ax[1].plot([mean[0] + std[0], mean[0] + std[0]], [-extrema, extrema], color='tab:blue', linestyle='--', label='Std Deviation', alpha=0.5, linewidth=2)

        ax[1].plot([-extrema, extrema], [mean[1], mean[1]], color='tab:blue', linestyle='--', linewidth=2)
        ax[1].plot([-extrema, extrema], [mean[1] - std[1], mean[1] - std[1]], color='tab:blue', linestyle='--', alpha=0.5, linewidth=2)
        ax[1].plot([-extrema, extrema], [mean[1] + std[1], mean[1] + std[1]], color='tab:blue', linestyle='--', alpha=0.5, linewidth=2)

        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]

        ax[1].scatter(x, y, s=100, c='black')

        fig.legend(handles=[median_line, mean_line, std_line], loc='lower right', bbox_to_anchor=(0.98, 0.1))
        fig.tight_layout()
        fig.savefig(join(to_save, file_name))
        plt.close()

def plot_mean_density(data : pd.DataFrame, path : str, file_name = 'mean_density.png', models : Union[tuple, str] = 'all'):
    """ Plot the figure with the mean dash lines for the models and the density.
        
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'mean_density.png'.
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
        
        ax.set_xlabel(f'Errors of model 1')
        ax.set_ylabel(f'Errors of model 2')

        fig.legend(handles=[mean_line, std_line], loc='lower right', bbox_to_anchor=(0.84, 0.14))
        # fig.tight_layout()
        fig.savefig(join(to_save, file_name))
        plt.close()

def plot_hourglass(data : pd.DataFrame, path : str, file_name = 'hourglass.png', models : Union[tuple, str] = 'all'):
    """ Plot the figure only with the hourglass for the models
        
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'hourglass.png'.
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
        equal_points, = ax.plot([-extrema, extrema], [-extrema, extrema], label="Equal errors")
        ax.plot([-extrema, extrema], [extrema, -extrema], color='tab:blue', linewidth=1)

        # Model in abs is better
        abs_better, _ = ax.fill(
            [-extrema, 0, extrema], [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema],  
            c='tab:orange', 
            alpha=0.2, 
            label=f'Model 1 is better')
        # Model in ord is better
        ord_better, _ = ax.fill(
            [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema], [-extrema, 0, extrema], 
            c='tab:green', 
            alpha=0.2, 
            label=f'Model 2 is better')
        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]

        ax.scatter(x, y, s=100, c='black', alpha=0.9)

        ax.set_xlabel(f'Errors of model 1')
        ax.xaxis.label.set_color('tab:orange')
        ax.set_ylabel(f'Errors of model 2')
        ax.yaxis.label.set_color('tab:green')

        fig.tight_layout()
        fig.legend(handles=[abs_better, ord_better, equal_points], loc='lower right', bbox_to_anchor=(0.97, 0.09))
        fig.savefig(join(to_save, file_name))
        plt.close()

def plot_distributions_alone(data : pd.DataFrame, path : str, file_name = 'distribution.png', 
                             models : Union[tuple, str] = 'all', model_index = 0):
    """ Plot the figure with the distribution of errors for each model alone

    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'distribution.png'.
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
        hist = ax.hist(data['error_'+combination[model_index]], bins=50, alpha=0.8, edgecolor='black', color=color)
        ax.grid(True, linestyle='--', alpha=0.5)
        ticks = ax.get_yticks()
        ticks = [tick for tick in ticks if tick != 0]
        ax.set_yticks(ticks)

        # draw the gaussian kernel density
        # kde = gaussian_kde(data['error_'+combination[model_index]])
        # x = np.linspace(-extrema, extrema, 1000)
        # ax.plot(x, kde(x) * len(data) * (extrema * 2) / 50, color='tab:blue', label='KDE')

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
        fig.savefig(join(to_save, file_name))

def plot_diff_distributions(data : pd.DataFrame, path : str, file_name = 'distributions_diff.png', 
                            models : Union[tuple, str] = 'all'):
    """ Plot the figure with the distribution of the differences between the errors for the models.

    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'distributions_diff.png'.
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
        extrema = max(abs(data['error_'+combination[0]] - data['error_'+combination[1]]))
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharex=True, sharey=True)

        hist = ax.hist(data['error_'+combination[0]] - data['error_'+combination[1]], bins=50, alpha=0.8, edgecolor='black', color='tab:orange')
        ax.grid(True, linestyle='--', alpha=0.5)
        ticks = ax.get_yticks()
        ticks = [tick for tick in ticks if tick != 0]
        ax.set_yticks(ticks)

        # draw the gaussian kernel density
        # kde = gaussian_kde(data['error_'+model])
        # x = np.linspace(-extrema, extrema, 1000)
        # ax.plot(x, kde(x) * len(data) * (extrema * 2) / 50, color='tab:blue', label='KDE')

        ax.set_xlim(-extrema, extrema)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Errors difference')

        max_bin_height = max(hist[0])
        max_bin_index = np.where(hist[0] == max_bin_height)[0][0]
        mode = (hist[1][max_bin_index] + hist[1][max_bin_index + 1]) / 2

        ax.plot([mode, mode], [0, max_bin_height], color='tab:blue', linestyle='--', label=f'Mode')
        ax.text(mode, max_bin_height, f'{mode:.2f}', ha='center', va='bottom', color='tab:blue', fontsize=15)

        ax.legend()
        fig.tight_layout()
        fig.savefig(join(to_save, file_name))

def plot_distributions(data : pd.DataFrame, path : str, file_name = 'distributions.png', models : Union[tuple, str] = 'all'):
    """ Plot the figure with the distributions of the errors for the models
        
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'distributions.png'.
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
        fig, ax = plt.subplots(2, 1, figsize=(16, 14), sharex=True, sharey=True)
        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]

        hist1 = ax[0].hist(x, bins=50, alpha=0.8, label='Model 1', edgecolor='black', color='tab:orange')
        ax[0].grid(True, linestyle='--', alpha=0.5)
        ax[0].set_xlim(-extrema, extrema)
        ticks = [tick for tick in ax[0].get_yticks() if tick != 0]
        ax[0].set_yticks(ticks)

        hist2 = ax[1].hist(y, bins=50, alpha=0.5, label='Model 2', edgecolor='black', color='tab:green')
        ax[1].grid(True, linestyle='--', alpha=0.5)
        ax[1].set_xlim(-extrema, extrema)
        ticks = [tick for tick in ax[1].get_yticks() if tick != 0]
        ax[1].set_yticks(ticks)

        max_bin_height = max(hist1[0])
        max_bin_index = np.where(hist1[0] == max_bin_height)[0][0]
        mode1 = (hist1[1][max_bin_index] + hist1[1][max_bin_index + 1]) / 2
        ax[0].plot([mode1, mode1], [0, max_bin_height], color='tab:blue', linestyle='--')
        ax[0].text(mode1, max_bin_height, f'{mode1:.2f}', ha='center', va='bottom', color='tab:blue', fontsize=15)

        max_bin_height = max(hist2[0])
        max_bin_index = np.where(hist2[0] == max_bin_height)[0][0]
        mode2 = (hist2[1][max_bin_index] + hist2[1][max_bin_index + 1]) / 2
        ax[1].plot([mode2, mode2], [0, max_bin_height], color='tab:blue', linestyle='--', label=f'Mode')
        ax[1].text(mode2, max_bin_height, f'{mode2:.2f}', ha='center', va='bottom', color='tab:blue', fontsize=15)

        ax[0].set_ylabel('Frequency')
        ax[1].set_xlabel('Errors')
        ax[1].set_ylabel('Frequency')
        fig.legend(loc='upper right')
        fig.subplots_adjust(hspace=0)
        fig.tight_layout()
        fig.savefig(join(to_save, file_name))

def plot_with_proximity(
        data : pd.DataFrame, path : str, file_name = 'circle_plot.png', 
        models : Union[tuple, str] = 'all', colormap : str = 'Spectral',
        distance_metric : str = 'mahalanobis', with_hourglass : bool = True):
    """ Plot the figure with the proximity of the points.

    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save. Defaults to 'circle_plot.png'.
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    colormap (str, optional): The colormap to use. Defaults to 'Spectral'.
    distance_metric (str, optional): The distance to use. Defaults to 'mahalanobis'.
    with_hourglass (bool, optional): If True, plot the hourglass. Defaults to True.
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
        extrema = max(
            abs(data[['error_'+model for model in combination]].min().min()), 
            abs(data[['error_'+model for model in combination]].max().max()))
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        ax.set_xlim(-extrema, extrema)
        ax.set_ylim(-extrema, extrema)
        ax.set_aspect('equal', adjustable='box')

        # Vertical axis
        ax.plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        # Horizontal axis
        ax.plot([-extrema, extrema], [0, 0], color='black', linewidth=1)
        # Diagonal
        equal_points, = ax.plot([-extrema, extrema], [-extrema, extrema], label="Equal errors")
        ax.plot([-extrema, extrema], [extrema, -extrema], color='tab:blue', linewidth=1)

        if with_hourglass:
            abs_better, _ = ax.fill(
                [-extrema, 0, extrema], [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema],  
                c='tab:orange', 
                alpha=0.2, 
                label=f'Model 1 is better')
            ord_better, _ = ax.fill(
                [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema], [-extrema, 0, extrema], 
                c='tab:green', 
                alpha=0.2, 
                label=f'Model 2 is better')

        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]

        # Calculate distance to the median
        median = (data['error_'+combination[0]].median(), data['error_'+combination[1]].median())
        if distance_metric == 'euclidean':
            distance = np.sqrt((x - median[0])**2 + (y - median[1])**2)
        if distance_metric == 'manhattan':
            distance = np.abs(x - median[0]) + np.abs(y - median[1])
        else:
            cov = np.cov(data[['error_'+combination[0], 'error_'+combination[1]]], rowvar=False)
            # calculate coordinates for each point
            distance = []
            for row in data[['error_'+combination[0], 'error_'+combination[1]]].values:
                distance.append(mahalanobis(row, median, np.linalg.inv(cov)))
        data['distance'] = distance

        data = data.sort_values(by='distance')
        # Get for each point, the percentage of points that are at least that distance
        data['percentile'] = data['distance'].apply(lambda x: (len(data[data['distance'] <= x]) / len(data)) * 100)
        data = data.sort_index()

        density = ax.scatter(x, y, c=data['percentile'], s=100, cmap=colormap)
        fig.colorbar(density, label="Percentile", fraction=0.030)

        # draw a cross on the median point (median[0], median[1])
        ax.plot(median[0], median[1], 'x', color='black', markersize=5)

        ax.set_xlabel(f'Errors of {combination[0]}')
        ax.set_ylabel(f'Errors of {combination[1]}')
        if with_hourglass:
            ax.xaxis.label.set_color('tab:orange')
            ax.yaxis.label.set_color('tab:green')
            fig.legend(handles=[abs_better, ord_better], loc='lower right')
        else:
            fig.legend(handles=[equal_points], loc='lower right')
        fig.subplots_adjust(left=0.15)
        fig.savefig(join(to_save, file_name))
        plt.close()

def plot_density_proximity(data : pd.DataFrame, path : str, file_name = 'density_proximity.png',
                            models : Union[tuple, str] = 'all'):
    """ Plot the figure with the density vs proximity of the points.

    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save. Defaults to 'density_proximity.png'.
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
        fig, ax = plt.subplots(1, 2, figsize=(16, 16), sharex=True, sharey=True)
        # Density plot
        ax[0].set_xlim(-extrema, extrema)
        ax[0].set_ylim(-extrema, extrema)
        ax[0].set_aspect('equal', adjustable='box')
        # Vertical axis
        ax[0].plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        # Horizontal axis
        ax[0].plot([-extrema, extrema], [0, 0], color='black', linewidth=1)
        # Diagonal
        equal_points, = ax[0].plot([-extrema, extrema], [-extrema, extrema], label="Equal errors")
        ax[0].plot([-extrema, extrema], [extrema, -extrema], color='tab:blue', linewidth=1)

        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]

        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        new_x, new_y, new_z = x[idx], y[idx], z[idx]
        density = ax[0].scatter(new_x, new_y, c=new_z, s=100)
        fig.colorbar(density, label="KDE", fraction=0.030)

        ax[0].set_xlabel(f'Errors of {combination[0]}')
        ax[0].set_ylabel(f'Errors of {combination[1]}')

        # Proximity plot
        ax[1].set_xlim(-extrema, extrema)
        ax[1].set_ylim(-extrema, extrema)
        ax[1].set_aspect('equal', adjustable='box')

        # Vertical axis
        ax[1].plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
        # Horizontal axis
        ax[1].plot([-extrema, extrema], [0, 0], color='black', linewidth=1)
        # Diagonal
        equal_points, = ax[1].plot([-extrema, extrema], [-extrema, extrema], label="Equal errors")
        ax[1].plot([-extrema, extrema], [extrema, -extrema], color='tab:blue', linewidth=1)

        # Calculate distance to the median
        median = (data['error_'+combination[0]].median(), data['error_'+combination[1]].median())
        distance = np.sqrt((x - median[0])**2 + (y - median[1])**2)
        data['distance'] = distance

        data = data.sort_values(by='distance')
        # Get for each point, the percentage of points that are at least that distance
        data['percentile'] = data['distance'].apply(lambda x: (len(data[data['distance'] <= x]) / len(data)) * 100)
        data = data.sort_index()

        density = ax[1].scatter(x, y, c=data['percentile'], s=100, cmap='Spectral')
        fig.colorbar(density, label="Percentile", fraction=0.030)
        ax[1].plot(median[0], median[1], 'x', color='black', markersize=5)

        ax[1].set_xlabel(f'Errors of {combination[0]}')
        ax[1].set_ylabel(f'Errors of {combination[1]}')

        fig.tight_layout()
        fig.legend(handles=[equal_points], loc='lower right', bbox_to_anchor=(0.97, 0.22))
        fig.savefig(join(to_save, file_name))
        plt.close()

def plot_compared_proximity(
        data : pd.DataFrame, path : str, file_name = 'circle_plot.png', 
        models : Union[tuple, str] = 'all', colormap : str = 'Spectral',
        with_hourglass : bool = True):
    """ Plot a comparison of two distance metrics for the models.

    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save. Defaults to 'circle_plot.png'.
    models (Union[tuple, str], optional): The models to plot. If 'all', plots all combinations of error metrics. Defaults to 'all'.
    colormap (str, optional): The colormap to use. Defaults to 'Spectral'.
    with_hourglass (bool, optional): If True, plot the hourglass. Defaults to True.
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
        extrema = max(
            abs(data[['error_'+model for model in combination]].min().min()), 
            abs(data[['error_'+model for model in combination]].max().max()))
        fig, ax = plt.subplots(1, 2, figsize=(16, 16), sharex=True, sharey=True)
        for axis in ax:
            axis.set_xlim(-extrema, extrema)
            axis.set_ylim(-extrema, extrema)
            axis.set_aspect('equal', adjustable='box')

            # Vertical axis
            axis.plot([0, 0], [-extrema, extrema], color='black', linewidth=1)
            # Horizontal axis
            axis.plot([-extrema, extrema], [0, 0], color='black', linewidth=1)
            # Diagonal
            equal_points, = axis.plot([-extrema, extrema], [-extrema, extrema], label="Equal errors")
            axis.plot([-extrema, extrema], [extrema, -extrema], color='tab:blue', linewidth=1)

            if with_hourglass:
                abs_better, _ = axis.fill(
                    [-extrema, 0, extrema], [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema],  
                    c='tab:orange', 
                    alpha=0.2, 
                    label=f'Model 1 is better')
                ord_better, _ = axis.fill(
                    [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema], [-extrema, 0, extrema], 
                    c='tab:green', 
                    alpha=0.2, 
                    label=f'Model 2 is better')

        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]

        # Calculate distance to the median
        median = (data['error_'+combination[0]].median(), data['error_'+combination[1]].median())
        distance1 = np.sqrt((x - median[0])**2 + (y - median[1])**2)
        cov = np.cov(data[['error_'+combination[0], 'error_'+combination[1]]], rowvar=False)
        # calculate coordinates for each point
        distance2 = []
        for row in data[['error_'+combination[0], 'error_'+combination[1]]].values:
            distance2.append(mahalanobis(row, median, np.linalg.inv(cov)))
        data['distance1'] = distance1
        data['distance2'] = distance2

        for i in range(2):
            data = data.sort_values(by=f'distance{i+1}')
            # Get for each point, the percentage of points that are at least that distance
            data['percentile'] = data[f'distance{i+1}'].apply(lambda x: (len(data[data[f'distance{i+1}'] <= x]) / len(data)) * 100)
            data = data.sort_index()

            density = ax[i].scatter(x, y, c=data['percentile'], s=100, cmap=colormap)

        cbar_ax = fig.add_axes([0.15, 0.2, 0.7, 0.02])
        fig.colorbar(density, cax=cbar_ax, orientation='horizontal', label="Percentile")

        fig.text(0.5, 0.27, f'Errors of {combination[0]}', ha='center', va='center', fontsize=25, color='tab:orange')
        ax[0].set_ylabel(f'Errors of {combination[1]}')
        if with_hourglass:
            for axis in ax:
                axis.xaxis.label.set_color('tab:orange')
                axis.yaxis.label.set_color('tab:green')
                axis.plot(median[0], median[1], 'x', color='black', markersize=5)
            fig.legend(handles=[abs_better, ord_better, equal_points], loc='upper right', bbox_to_anchor=(0.97, 0.8))
        else:
            fig.legend(handles=[equal_points], loc='upper right', bbox_to_anchor=(0.97, 0.7))
        fig.savefig(join(to_save, file_name))
        plt.close()

def plot_everything(data : pd.DataFrame, path : str, file_name = 'general_plot.png', 
                    models : Union[tuple, str] = 'all', show_one_individual = False, 
                    distance_metric : str = 'euclidean', colormap : str = 'Spectral'):
    """ Plot everything in one figure for the models.
        
    Parameters:
    data (pd.DataFrame): The input data containing the actual and predicted values.
    path (str): The path to save the generated plot(s).
    file_name (str, optional): The name of the file to save the plot. Defaults to 'general_plot.png'.
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
        equal_points, = ax.plot([-extrema, extrema], [-extrema, extrema], label="Equal errors")
        ax.plot([-extrema, extrema], [extrema, -extrema], color='tab:blue', linewidth=1)

        # Model in abs is better
        abs_better, _ = ax.fill(
            [-extrema, 0, extrema], [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema],  
            c='tab:orange', 
            alpha=0.2, 
            label=f'Model 1 is better')
        # Model in ord is better
        ord_better, _ = ax.fill(
            [-extrema, 0, -extrema], [-extrema, 0, extrema], [extrema, 0, extrema], [-extrema, 0, extrema], 
            c='tab:green', 
            alpha=0.2, 
            label=f'Model 2 is better')
        x = data['error_'+combination[0]]
        y = data['error_'+combination[1]]
        # xy = np.vstack([x, y])
        # # Order the points by density
        # z = gaussian_kde(xy)(xy)
        # idx = z.argsort()
        # x, y, z = x[idx], y[idx], z[idx]
        # density = ax.scatter(x, y, c=z, s=100)
        # fig.colorbar(density, label="KDE", fraction=0.030)

        # Calculate distance to the median
        median = (data['error_'+combination[0]].median(), data['error_'+combination[1]].median())
        if distance_metric == 'manhattan':
            distance = np.abs(x - median[0]) + np.abs(y - median[1])
        elif distance_metric == 'mahalanobis':
            cov = np.cov(data[['error_'+combination[0], 'error_'+combination[1]]], rowvar=False)
            distance = data[['error_'+combination[0], 'error_'+combination[1]]].apply(lambda x: mahalanobis(x, median, cov), axis=1)
        else:
            distance = np.sqrt((x - median[0])**2 + (y - median[1])**2)
        data['distance'] = distance

        data = data.sort_values(by='distance')
        # Get for each point, the percentage of points that are at least that distance
        data['percentile'] = data['distance'].apply(lambda x: (len(data[data['distance'] <= x]) / len(data)) * 100)
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

        # draw a cross on the median point (median[0], median[1])
        ax.plot(median[0], median[1], 'x', color='black', markersize=5)

        ax.set_xlabel(f'Errors of model 1')
        ax.xaxis.label.set_color('tab:orange')
        ax.set_ylabel(f'Errors of model 2')
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
        fig.savefig(join(to_save, file_name))
        plt.close()
