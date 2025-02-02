"""
These are utility functions for Optuna Hyperparameter optimization. 
The code supports generating and saving visualizations of the optimization process using both Plotly and Matplotlib.
"""

import os
import datetime
import matplotlib.pyplot as plt
import optuna

def make_dirs(dirpath):
    """
    Creates a directory if it does not already exist.

    Args:
        - dirpath (str): The path of the directory to be created.
    """
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

def add_suffix(file, suffix):
    """
    Adds a suffix to the filename before the extension.

    Args:
        - file (str): The original filename.
        - suffix (str): The suffix to be added to the filename.

    Output:
        - str: The modified filename with the added suffix.
    """
    root, ext = os.path.splitext(file)
    return root + suffix + ext

def write_image(ax, dirname, filename, suffix='_ply'):
    """
    Writes a Plotly image to a file.

    Args:
        - ax: The Plotly figure object.
        - dirname (str): Directory where the image will be saved.
        - filename (str): Name of the image file.
        - suffix (str): Suffix to be added to the filename.
    """
    filename = add_suffix(filename, suffix)
    filepath = os.path.join(dirname, filename)
    ax.write_image(filepath)

def savefig(dirname, filename, suffix='_plt', dpi=100):
    """
    Saves a Matplotlib figure to a file.

    Args:
        - dirname (str): Directory where the image will be saved.
        - filename (str): Name of the image file.
        - suffix (str): Suffix to be added to the filename.
        - dpi (int): Resolution of the saved figure.
    """
    filename = add_suffix(filename, suffix)
    filepath = os.path.join(dirname, filename)
    plt.savefig(filepath, dpi=dpi)

def get_all_params(study):
    """
    Retrieves all unique parameter names from completed trials in the study.

    Args:
        - study: An Optuna study object.

    Output:
        - list or None: A sorted list of unique parameter names or None if no completed trials are found.
    """
    trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(trials) == 0:
        return None

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    return sorted(list(all_params))

def output_results_plotly(study, all_params, output_dirname, max_resolution_rate, output_type='maximum'):
    """
    Generates and saves Plotly visualizations for the optimization process.

    Args:
        - study: An Optuna study object.
        - all_params (list): A list of all parameter names.
        - output_dirname (str): Directory where the output images will be saved.
        - max_resolution_rate (int): The maximum resolution scaling factor.
        - output_type (str): The type of visualizations to generate.
    """
    print('executing plot_optimization_history...')
    ax = optuna.visualization.plot_optimization_history(study)
    write_image(ax, output_dirname, 'optimization_history.png')

    if output_type in ['slice', 'standard', 'maximum']:
        print('executing plot_parallel_coordinate...')
        ax = optuna.visualization.plot_parallel_coordinate(study)
        write_image(ax, output_dirname, 'parallel_coordinate.png')

    if output_type in ['maximum']:
        print('executing plot_contour...')
        ax = optuna.visualization.plot_contour(study)
        for rate in range(1, max_resolution_rate + 1):
            ax.update_layout(width=int(700 * rate), height=int(500 * rate))
            write_image(ax, output_dirname, 'contour.png', suffix='_ply_' + str(rate))

    if output_type in ['slice', 'standard', 'maximum']:
        print('executing plot_slice...')
        output_dirname_tmp = os.path.join(output_dirname, 'slice_ply')
        make_dirs(output_dirname_tmp)
        for param1 in all_params:
            ax = optuna.visualization.plot_slice(study, params=[param1])
            write_image(ax, output_dirname_tmp, param1 + '.png')

        print('executing plot_param_importances...')
        ax = optuna.visualization.plot_param_importances(study)
        write_image(ax, output_dirname, 'param_importances.png')

        print('executing plot_edf...')
        ax = optuna.visualization.plot_edf(study)
        write_image(ax, output_dirname, 'edf.png')

def output_results_pyplot(study, all_params, output_dirname, max_resolution_rate, output_type='maximum'):
    """
    Generates and saves Matplotlib visualizations for the optimization process.

    Args:
        - study: An Optuna study object.
        - all_params (list): A list of all parameter names.
        - output_dirname (str): Directory where the output images will be saved.
        - max_resolution_rate (int): The maximum resolution scaling factor.
        - output_type (str): The type of visualizations to generate.
    """
    print('executing matplotlib.plot_optimization_history...')
    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    savefig(output_dirname, 'optimization_history.png')

    if output_type in ['slice', 'standard', 'maximum']:
        print('executing matplotlib.plot_parallel_coordinate...')
        ax = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        fig = plt.gcf()
        for rate in range(1, max_resolution_rate + 1):
            fig.set_size_inches(6.4 * rate, 4.8 * rate)
            savefig(output_dirname, 'parallel_coordinate.png', suffix='_plt_' + str(rate), dpi=int(100 * rate))

    print('executing matplotlib.plot_edf...')
    ax = optuna.visualization.matplotlib.plot_edf(study)
    savefig(output_dirname, 'edf.png')

def output_results(study, model_name, output_type='maximum', output_mode_list=['plotly', 'pyplot'], max_resolution_rate=3):
    """
    Outputs results and visualizations for an Optuna study.

    Args:
        - study: An Optuna study object.
        - model_name (str): Name of the model being optimized.
        - output_type (str): The type of visualizations to generate.
        - output_mode_list (list): List of output modes ('plotly' or 'pyplot').
        - max_resolution_rate (int): The maximum resolution scaling factor.
    """
    if output_type in ['slice', 'standard', 'maximum']:
        print('best params : {}'.format(study.best_params))
        print('best trial : {}'.format(study.best_trial))
        print('trials_dataframe : {}'.format(study.trials_dataframe()))

    output_dirname = model_name + '_summary'
    make_dirs(output_dirname)

    print('executing trials_dataframe...')
    study.trials_dataframe().to_csv(os.path.join(output_dirname, 'trials.csv'))

    all_params = get_all_params(study)

    if 'plotly' in output_mode_list:
        output_results_plotly(study, all_params, output_dirname, max_resolution_rate, output_type=output_type)

    if 'pyplot' in output_mode_list:
        output_results_pyplot(study, all_params, output_dirname, max_resolution_rate, output_type=output_type)
