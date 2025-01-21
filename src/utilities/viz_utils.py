import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX."""
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)

def plot_training_loss(avg_regrets, base_dir, window_size=128, file_name="moving_average_plot.pdf", 
                       xlabel='Step', ylabel='Moving Average of Max Regret', title=None,
                       exp_fit_range=[140, 200], power_fit_range=[200, None]):
    """
    Plots the moving average of the maximum values along axis=1 of the avg_regrets array,
    including exponential and power-law fits with confidence intervals.
    
    Args:
        avg_regrets: Array of average regrets with shape (n, 2).
        window_size: The size of the window for computing the moving average.
        base_dir: Directory to save the plot.
        file_name: Name of the file to save the plot as.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        title: Title of the plot.
        exp_fit_range: Range (as [min, max]) for the exponential fit.
        power_fit_range: Range (as [min, max]) for the power-law fit (use None for max to fit until end).
    """
    
    # Compute the maximum along axis=1
    max_regrets = np.max(avg_regrets, axis=1)
    #idxs = np.arange(len(max_regrets))
    
    # Compute the moving average
    #moving_avg = np.convolve(max_regrets, np.ones(window_size)/window_size, mode='valid')
    
    # undersanple
    idxs = np.logspace(np.log10(1), np.log10(len(max_regrets)), num=256, dtype=int, endpoint=True)-1
    max_regrets = max_regrets[idxs]
    
    # Plot the moving average
    fig_width, fig_height = set_size(452.9679, fraction=0.5)  # Adjust 'fraction' as needed
    plt.figure(figsize=(fig_width, fig_height))
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 11,
        "font.size": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": 'black',
        'axes.linewidth': 0.5
    })
    
    plt.plot(idxs, max_regrets, color='blue', linewidth=1.25)
    plt.xscale('log', base=10)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(visible=True, color='grey', linestyle='-', linewidth=0.25, alpha=0.2)
    
    if title:
        plt.title(title, fontsize=11)
        
    plot_file_name = os.path.join(base_dir, file_name)
    plt.savefig(plot_file_name, format='pdf', bbox_inches='tight')
    plt.close()

def plot_histogram(data, base_dir, xlim=(0, 0.1), bins=50, file_name="histogram.png", xlabel='Values', ylabel='Frequency', title='Histogram of Values', color='blue'):
    """
    Plot a histogram of the data.

    Args:
        data (np.array): Array containing data values.
        base_dir (str): Base directory to save the plot file.
        xlim (tuple): Tuple containing lower and upper bounds of x-axis (default: (0, 0.1)).
        bins (int): Number of bins in the histogram (default: 50).
        file_name (str): Name of the plot file (default: "histogram.png").
        xlabel (str): Label for the x-axis (default: 'Values').
        ylabel (str): Label for the y-axis (default: 'Frequency').
        title (str): Title of the plot (default: 'Histogram of Values').
        color (str): Color of the histogram bars (default: 'blue').
    """
    plot_file_name = os.path.join(base_dir, file_name)

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    xlim=(0, np.quantile(data,0.95))
    plt.hist(data, bins=bins, range=xlim, edgecolor='black', color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xlim(xlim)

    # Save the plot to a file
    plt.savefig(plot_file_name)
    plt.close()


def plot_cdfs(data1, data2, base_dir, file_name="figure.pdf", xlabel='', ylabel='', 
              label1='', label2='', ylim_left = 0, title = ''):
    loss_data1 = data1.astype(np.float64)
    loss_data2 = data2.astype(np.float64)

    if data1.size == 0 or data2.size == 0:
        return

    # Sort the loss data and compute the CDFs
    sorted_data1 = np.sort(loss_data1)
    cdf1 = np.arange(1, len(sorted_data1) + 1) / len(sorted_data1)
    sorted_data2 = np.sort(loss_data2)
    cdf2 = np.arange(1, len(sorted_data2) + 1) / len(sorted_data2)

    # Find the values corresponding to the 99.3th percentile for both datasets
    xlim_right = max(np.percentile(loss_data1, 99.3), np.percentile(loss_data2, 99.3))

    # Find the values corresponding to the 99th percentile for both datasets
    value_99_data1 = np.percentile(loss_data1, 99)
    value_99_data2 = np.percentile(loss_data2, 99)

    # Find the corresponding CDF values
    cdf_99_data1 = np.interp(value_99_data1, sorted_data1, cdf1)
    cdf_99_data2 = np.interp(value_99_data2, sorted_data2, cdf2)

    # Set figure size using LaTeX text width
    fig_width, fig_height = set_size(452.9679, fraction=0.5)
    plt.figure(figsize=(fig_width, fig_height))

    # Use LaTeX for text rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 11,       # Match document font size
        "font.size": 11,            # Match document font size
        "legend.fontsize": 9,       # Slightly smaller than the main text
        "xtick.labelsize": 9,       # Slightly smaller than the main text
        "ytick.labelsize": 9,       # Slightly smaller than the main text
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": 'black',
        'axes.linewidth': 0.5
    })

    # Plot the CDFs for both datasets
    plt.step(sorted_data1, cdf1, where='post', color='red', linewidth=1.5, label=label1)
    plt.plot(value_99_data1, cdf_99_data1, 'ro', markersize=3)

    plt.step(sorted_data2, cdf2, where='post', color='blue', linewidth=1.5, label=label2)
    plt.plot(value_99_data2, cdf_99_data2, 'bo', markersize=3)

    plt.tick_params(direction="in", color='grey', width=0.25)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(visible=True, color='grey', linestyle='-', linewidth=0.25, alpha=0.2)
    plt.xlim(-xlim_right/64.72, xlim_right)
    plt.ylim(ylim_left - (1 - ylim_left)/40, 1 + (1 - ylim_left)/20)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
    plt.legend().get_frame().set_linewidth(0.55)
    plt.title(title,  fontsize=11)
    
    # Ensure the figures directory exists
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plot_file_name = os.path.join(figures_dir, file_name)
    plt.savefig(plot_file_name, format='pdf', bbox_inches='tight')
    plt.close()

def plot_learning_curves(max_regret, statistics, model_log_steps, base_dir, 
                         file_name="avg_regrets_plot.pdf", xlabel='Step', ylabel='Average Regret', 
                         title=None, legend_labels=None, confidence = 0.9):
    """
    Plots average regrets with options for customization of labels, title, legend, and confidence intervals.

    Args:
        max_regret: Array of max_regret.
        statistics: Dictionary containing statistical information.
        model_log_steps: Array of log-scaled steps.
        base_dir: Directory to save the plot.
        file_name: Name of the file to save the plot as.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        title: Title of the plot.
        legend_labels: List of labels for the legend.
    """
    
    # Compute masks and averages
    zero_pure_nash_mask = statistics['n_pure_nash'] == 0
    some_pure_nash_mask = statistics['n_pure_nash'] > 0
    
    avg_regrets = max_regret.mean(axis=1)
    zero_pure_avg_regrets = max_regret[:, zero_pure_nash_mask].mean(axis=1)
    some_pure_avg_regrets = max_regret[:, some_pure_nash_mask].mean(axis=1)
    
    # Apply mask for steps greater than or equal to 100
    mask = model_log_steps >= 100
    idxs = model_log_steps[mask]
    avg_regrets = avg_regrets[mask]
    zero_pure_avg_regrets = zero_pure_avg_regrets[mask]
    some_pure_avg_regrets = some_pure_avg_regrets[mask]
    
    # Compute quantile-based confidence intervals
    def compute_quantile_intervals(data):
        lower_quantile = np.quantile(data, (1 - confidence)/2, axis=1)
        upper_quantile = np.quantile(data, (1 + confidence)/2, axis=1)
        return lower_quantile, upper_quantile
    
    ci_zero_pure_lower, ci_zero_pure_upper = compute_quantile_intervals(max_regret[:, zero_pure_nash_mask])
    ci_some_pure_lower, ci_some_pure_upper = compute_quantile_intervals(max_regret[:, some_pure_nash_mask])
    
    ci_zero_pure_lower = ci_zero_pure_lower[mask]
    ci_zero_pure_upper = ci_zero_pure_upper[mask]
    ci_some_pure_lower = ci_some_pure_lower[mask]
    ci_some_pure_upper = ci_some_pure_upper[mask]
    
    # Set figure size using LaTeX text width
    fig_width, fig_height = set_size(452.9679, fraction=0.5)
    plt.figure(figsize=(fig_width, fig_height))
    
    # Use LaTeX for text rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 11,       # Match document font size
        "font.size": 11,            # Match document font size
        "legend.fontsize": 9,       # Slightly smaller than the main text
        "xtick.labelsize": 9,       # Slightly smaller than the main text
        "ytick.labelsize": 9,       # Slightly smaller than the main text
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": 'black',
        'axes.linewidth': 0.5
    })
        
    # Plot the data
    labels = legend_labels if legend_labels else [r'$>0$ PURE', r'$0$ PURE', 'ALL GAMES']
    plt.plot(idxs, some_pure_avg_regrets, color='red', label=labels[0], linewidth=1)
    plt.plot(idxs, zero_pure_avg_regrets, color='blue', label=labels[1], linewidth=1)
    #plt.plot(idxs, avg_regrets, label=labels[2], linewidth=1)
    
    # Plot quantile-based confidence intervals as shaded areas
    plt.fill_between(idxs, ci_some_pure_lower, ci_some_pure_upper, color='red', alpha=0.075, edgecolor=None)
    plt.fill_between(idxs, ci_zero_pure_lower, ci_zero_pure_upper, color='blue', alpha=0.075, edgecolor=None)
    
    # Apply log scale to the x-axis
    plt.xscale('log', base=10)
    
    # Set labels and grid
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(visible=True, color='grey', linestyle='-', linewidth=0.25, alpha=0.2)
    
    # Set the legend
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.legend().get_frame().set_linewidth(0.55)
    
    # Set the title if provided
    if title:
        plt.title(title, fontsize=11)
    
    # Save the plot as a PDF file
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plot_file_name = os.path.join(figures_dir, file_name)
    plt.savefig(plot_file_name, format='pdf', bbox_inches='tight')
    plt.close()
