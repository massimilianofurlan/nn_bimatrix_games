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

def set_style(plt):
    """Sets matplotlib rcParams"""
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

def plot_total_regret(avg_regrets, base_dir, file_name="learning_curve.pdf", 
                      xlabel='Log Interval', ylabel='Total Regret'):
    """
    Plots sum of regrets during training (mean over 128 intervals)
    """
    
    # Compute the maximum along axis=1
    max_regrets = np.sum(avg_regrets, axis=1)
    idxs = np.arange(1, len(max_regrets)+1)
    
    # Plot the moving average
    fig_width, fig_height = set_size(452.9679, fraction=0.5)  # Adjust 'fraction' as needed
    plt.figure(figsize=(fig_width, fig_height))
    set_style(plt)
    
    plt.plot(idxs, max_regrets, color='blue', linewidth=1.25)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(visible=True, color='grey', linestyle='-', linewidth=0.25, alpha=0.2)
    
    plot_file_name = os.path.join(base_dir, file_name)
    plt.savefig(plot_file_name, format='pdf', bbox_inches='tight')
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
    set_style(plt)
    
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
    """
    
    # Compute masks and averages
    zero_pure_nash_mask = statistics['n_pure_nash'] == 0
    some_pure_nash_mask = statistics['n_pure_nash'] > 0
    
    avg_regrets = max_regret.mean(axis=1)
    zero_pure_avg_regrets = max_regret[:, zero_pure_nash_mask].mean(axis=1)
    some_pure_avg_regrets = max_regret[:, some_pure_nash_mask].mean(axis=1)
    
    # Apply mask for steps greater than or equal to 100
    mask = model_log_steps >= 50
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
    set_style(plt)
        
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

def plot_learning_curve_fit(regret_profiles, model_log_steps, base_dir, file_name="learning_curve_fit.pdf", 
                            xlabel='Step', ylabel='MaxReg', title=None,   
                            exp_fit_range=[150, 180], power_fit_range=[180, None]):
    """
    Plots average max regret across games and fits exponential and power-law decay curves.
    """
    
    # Compute average max regret across games
    avg_max_regret = regret_profiles.max(axis=2).mean(axis=1)
    
    # Truncate early steps
    mask = model_log_steps >= 50
    model_log_steps = model_log_steps[mask]
    avg_max_regret = avg_max_regret[mask]
    
    # Set figure size and style
    fig_width, fig_height = set_size(452.9679, fraction=0.50)
    plt.figure(figsize=(fig_width, fig_height))
    set_style(plt)
    
    # Plot learning curve
    plt.plot(model_log_steps, avg_max_regret, color='blue', linewidth=1.25)
    plt.xscale('log', base=10)
    
    A_exp = b_exp = A_power = b_power = None
    
    # === Exponential Fit ===
    exp_mask = (model_log_steps >= exp_fit_range[0]) & (model_log_steps <= exp_fit_range[1])
    exp_x = model_log_steps[exp_mask]
    exp_y = avg_max_regret[exp_mask]
    if len(exp_x) > 0:
        def exp_decay(x, A, b):
            return A * np.exp(-b * x)
        popt_exp, _ = curve_fit(exp_decay, exp_x, exp_y, p0=(exp_y[0], 0.01))
        A_exp, b_exp = popt_exp
        y_fit_exp = exp_decay(exp_x, A_exp, b_exp)
        plt.plot(exp_x, y_fit_exp, linestyle='--', color='limegreen', linewidth=0.75)
    
    # === Power-Law Fit ===
    max_x = power_fit_range[1] if power_fit_range[1] is not None else model_log_steps[-1]
    power_mask = (model_log_steps >= power_fit_range[0]) & (model_log_steps <= max_x)
    power_x = model_log_steps[power_mask]
    power_y = avg_max_regret[power_mask]
    if len(power_x) > 0:
        def power_law(x, A, b):
            return A * x**b
        popt_power, _ = curve_fit(power_law, power_x, power_y)
        A_power, b_power = popt_power
        y_fit_power = power_law(power_x, A_power, b_power)
        plt.plot(power_x, y_fit_power, linestyle='--', color='red', linewidth=0.75)
    
    # Final styling
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title, fontsize=11)
    plt.grid(visible=True, color='grey', linestyle='-', linewidth=0.25, alpha=0.2)
    
    # Save plot
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plot_file_path = os.path.join(figures_dir, file_name)
    plt.savefig(plot_file_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    # Save fit details to .txt
    fit_txt_file = os.path.splitext(file_name)[0] + '.txt'
    fit_txt_path = os.path.join(figures_dir, fit_txt_file)
    with open(fit_txt_path, 'w') as f:
        f.write(f"# Exponential Fit: A * exp(-b * x)\n")
        f.write(f"A = {A_exp:.6f}\n")
        f.write(f"b = {b_exp:.6f}\n")
        f.write("\n")
        f.write(f"# Power Law Fit: A * x^b\n")
        f.write(f"A = {A_power:.6f}\n")
        f.write(f"b = {b_power:.6f}\n")

