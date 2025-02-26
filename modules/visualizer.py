import matplotlib.pyplot as plt
import numpy as np

def plot_survival_curve(results, show_ci=True, ci_level=95):
    """
    Plot the survival function curve with confidence intervals
    
    Parameters:
    -----------
    results : DataFrame
        DataFrame with survival analysis results
    show_ci : bool
        Whether to show confidence intervals
    ci_level : int
        Confidence level (percentage)
        
    Returns:
    --------
    matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract time points for plotting (use the end of each interval)
    time_points = [float(interval.split('-')[1]) for interval in results['interval']]
    
    # Add time point 0 for complete curve (S(0) = 1)
    time_points_with_zero = [0] + time_points
    survival_with_one = [1] + list(results['surv'])
    
    # Plot step function
    ax.step(time_points_with_zero, survival_with_one, where='post', 
           color='royalblue', linewidth=2, label='Survival Function')
    
    # Add confidence intervals if requested
    if show_ci:
        z = 1.96  # For 95% CI
        if ci_level != 95:
            from scipy.stats import norm
            z = norm.ppf(1 - (1 - ci_level/100)/2)
        
        lower_ci = [max(0, s - z * se) for s, se in zip(results['surv'], results['se_surv'])]
        upper_ci = [min(1, s + z * se) for s, se in zip(results['surv'], results['se_surv'])]
        
        # Add CI=1 at time 0
        lower_ci_with_one = [1] + lower_ci
        upper_ci_with_one = [1] + upper_ci
        
        ax.fill_between(time_points_with_zero, lower_ci_with_one, upper_ci_with_one,
                      step='post', alpha=0.2, color='royalblue',
                      label=f'{ci_level}% Confidence Interval')
    
    # Format the plot
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Estimated Survival Function', fontsize=14)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, max(time_points) * 1.05)
    ax.set_ylim(0, 1.05)
    
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    return fig

def plot_hazard_function(results):
    """
    Plot the hazard function
    
    Parameters:
    -----------
    results : DataFrame
        DataFrame with survival analysis results
        
    Returns:
    --------
    matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract midpoints for plotting
    interval_starts = [float(i.split('-')[0]) for i in results['interval']]
    interval_ends = [float(i.split('-')[1]) for i in results['interval']]
    midpoints = [(s + e) / 2 for s, e in zip(interval_starts, interval_ends)]
    
    # Plot hazard function as bar chart
    ax.bar(midpoints, results['hazard'], width=0.8, alpha=0.7, color='firebrick', 
          edgecolor='darkred', label='Hazard Function')
    
    # Add error bars if available
    if 'se_hazard' in results.columns:
        ax.errorbar(midpoints, results['hazard'], yerr=1.96 * results['se_hazard'],
                  fmt='none', ecolor='black', capsize=3, alpha=0.5)
    
    # Format the plot
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Hazard Rate', fontsize=12)
    ax.set_title('Estimated Hazard Function', fontsize=14)
    
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_xlim(0, max(interval_ends) * 1.05)
    
    # Set reasonable y-limits (filter out infinite values)
    finite_hazards = [h for h in results['hazard'] if not np.isinf(h) and not np.isnan(h)]
    if finite_hazards:
        ax.set_ylim(0, max(finite_hazards) * 1.2)
    
    plt.tight_layout()
    return fig

def plot_pdf(results):
    """
    Plot the probability density function
    
    Parameters:
    -----------
    results : DataFrame
        DataFrame with survival analysis results
        
    Returns:
    --------
    matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract midpoints for plotting
    interval_starts = [float(i.split('-')[0]) for i in results['interval']]
    interval_ends = [float(i.split('-')[1]) for i in results['interval']]
    midpoints = [(s + e) / 2 for s, e in zip(interval_starts, interval_ends)]
    
    # Plot PDF as a line with points
    ax.plot(midpoints, results['pdf'], 'o-', color='forestgreen', linewidth=2, 
           markersize=8, label='Probability Density Function')
    
    # Add area under the curve (approximation of the PDF)
    for i in range(len(midpoints)):
        if i < len(midpoints) - 1:
            # Width of each segment
            width = midpoints[i+1] - midpoints[i]
            # Height at midpoint
            height = results['pdf'].iloc[i]
            # Plot rectangle
            ax.add_patch(plt.Rectangle((midpoints[i] - width/2, 0), width, height,
                                     facecolor='forestgreen', alpha=0.1))
    
    # Format the plot
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Estimated Probability Density Function', fontsize=14)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, max(interval_ends) * 1.05)
    
    # Set reasonable y-limits
    ax.set_ylim(0, max(results['pdf']) * 1.2)
    
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig

def plot_cumulative_hazard(results):
    """
    Plot the cumulative hazard function
    
    Parameters:
    -----------
    results : DataFrame
        DataFrame with survival analysis results
        
    Returns:
    --------
    matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract time points for plotting
    time_points = [float(interval.split('-')[1]) for interval in results['interval']]
    
    # Calculate cumulative hazard: H(t) = -log(S(t))
    cum_hazard = [-np.log(surv) if surv > 0 else float('inf') for surv in results['surv']]
    
    # Add zero point
    time_points_with_zero = [0] + time_points
    cum_hazard_with_zero = [0] + cum_hazard
    
    # Plot step function
    ax.step(time_points_with_zero, cum_hazard_with_zero, where='post', 
           color='darkorange', linewidth=2, label='Cumulative Hazard')
    
    # Format the plot
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Cumulative Hazard', fontsize=12)
    ax.set_title('Estimated Cumulative Hazard Function', fontsize=14)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, max(time_points) * 1.05)
    
    # Set reasonable y-limits (filter out infinite values)
    finite_cum_hazard = [h for h in cum_hazard if not np.isinf(h) and not np.isnan(h)]
    if finite_cum_hazard:
        ax.set_ylim(0, max(finite_cum_hazard) * 1.2)
    
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig