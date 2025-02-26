import pandas as pd
import numpy as np
from scipy.stats import norm

def estimate_survival_function(data, sample_size=2418, alpha=0.05):
    """
    Estimate survival function from interval-censored data
    
    Parameters:
    -----------
    data : DataFrame
        Processed data with interval, deaths, and cens columns
    sample_size : int
        Initial sample size
    alpha : float
        Significance level for confidence intervals (default: 0.05)
        
    Returns:
    --------
    DataFrame with survival function estimates and related metrics
    """
    # Initialize results DataFrame
    results = pd.DataFrame()
    
    # Add interval column
    results['interval'] = data['interval']
    
    # Number at start of interval
    n_subs = [sample_size]
    for i in range(1, len(data)):
        n_subs.append(n_subs[i-1] - data['deaths'].iloc[i-1] - data['cens'].iloc[i-1])
    results['nsubs'] = n_subs
    
    # Number lost to censoring
    results['nlost'] = data['cens'].values
    
    # Number at risk (adjusted for censoring as per Dhafer Malouche's method)
    results['nrisk'] = [n - c/2 for n, c in zip(results['nsubs'], results['nlost'])]
    
    # Number of events (deaths)
    results['nevent'] = data['deaths'].values
    
    # Conditional probability of dying in the interval
    results['q'] = [d/r if r > 0 else 0 for d, r in zip(results['nevent'], results['nrisk'])]
    
    # Conditional probability of surviving the interval
    results['p'] = [1 - q for q in results['q']]
    
    # Cumulative survival function (product-limit estimator)
    results['surv'] = np.cumprod(results['p'])
    
    # Calculate interval midpoints and widths for hazard and density calculations
    interval_starts = [float(i.split('-')[0]) for i in results['interval']]
    interval_ends = [float(i.split('-')[1]) for i in results['interval']]
    midpoints = [(s + e) / 2 for s, e in zip(interval_starts, interval_ends)]
    widths = [e - s for s, e in zip(interval_starts, interval_ends)]
    
    # Hazard function (using the corrected formula from slide 7)
    results['hazard'] = [q / (w * (1 - q/2)) if q < 1 else float('inf') 
                        for q, w in zip(results['q'], widths)]
    
    # Probability density function (using formula from slide 8)
    # For first interval, use S(y_0) = 1
    pdf_values = []
    for i, (p, q, w) in enumerate(zip(results['p'], results['q'], widths)):
        if i == 0:
            s_prev = 1
        else:
            s_prev = results['surv'].iloc[i-1]
        pdf_values.append(s_prev * q / w)
    
    results['pdf'] = pdf_values
    
    # Standard errors for survival estimates
    # Using Greenwood's formula: Var[S(t)] = S(t)^2 * sum[d_i / (r_i * (r_i - d_i))]
    se_surv = []
    for i in range(len(results)):
        variance_sum = 0
        for j in range(i+1):
            nrisk = results['nrisk'].iloc[j]
            nevent = results['nevent'].iloc[j]
            if nrisk > nevent and nrisk > 0:
                variance_sum += nevent / (nrisk * (nrisk - nevent))
        se = results['surv'].iloc[i] * np.sqrt(variance_sum)
        se_surv.append(se)
    
    results['se_surv'] = se_surv
    
    # Calculate confidence intervals
    z = norm.ppf(1 - alpha/2)  # Z-score for the confidence level
    results['lower_ci'] = [max(0, s - z * se) for s, se in zip(results['surv'], results['se_surv'])]
    results['upper_ci'] = [min(1, s + z * se) for s, se in zip(results['surv'], results['se_surv'])]
    
    # Standard errors for hazard and PDF
    # Using delta method approximations
    results['se_hazard'] = [h / np.sqrt(d) if d > 0 else float('inf') 
                          for h, d in zip(results['hazard'], results['nevent'])]
    
    results['se_pdf'] = [pd / np.sqrt(d) if d > 0 else float('inf') 
                       for pd, d in zip(results['pdf'], results['nevent'])]
    
    return results