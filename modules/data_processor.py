import pandas as pd
import numpy as np
import streamlit as st

def load_data(file):
    """
    Load data from a CSV file
    
    Parameters:
    -----------
    file : uploaded file object
        CSV file uploaded through Streamlit
        
    Returns:
    --------
    DataFrame with the loaded data
    """
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def validate_data(df):
    """
    Validate the loaded data for required columns and format
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame to validate
        
    Returns:
    --------
    bool : True if valid, False otherwise
    """
    required_columns = ["intEndpoints", "deaths", "cens"]
    
    # Check for required columns
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing required columns. Expected: {required_columns}")
        return False
    
    # Check for numeric data types
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.error(f"Column '{col}' must contain numeric values")
            return False
    
    # Check for negative values
    if (df["deaths"] < 0).any() or (df["cens"] < 0).any():
        st.error("Deaths and censored observations cannot be negative")
        return False
    
    # Ensure intEndpoints are sorted
    if not df["intEndpoints"].is_monotonic_increasing:
        st.warning("Time endpoints are not in ascending order. Sorting automatically.")
        df.sort_values("intEndpoints", inplace=True)
    
    return True

def preprocess_data(df):
    """
    Preprocess the data for survival function estimation
    
    Parameters:
    -----------
    df : DataFrame
        Raw data with intEndpoints, deaths, and cens columns
        
    Returns:
    --------
    Processed DataFrame ready for analysis
    """
    processed_df = df.copy()
    
    # Ensure proper format for the results
    processed_df["intEndpoints"] = processed_df["intEndpoints"].astype(int)
    processed_df["deaths"] = processed_df["deaths"].astype(int)
    processed_df["cens"] = processed_df["cens"].astype(int)
    
    # Sort by interval endpoints if not already sorted
    processed_df = processed_df.sort_values("intEndpoints").reset_index(drop=True)
    
    # Create interval column for better readability
    intervals = []
    for i in range(len(processed_df) - 1):
        intervals.append(f"{processed_df['intEndpoints'].iloc[i]}-{processed_df['intEndpoints'].iloc[i+1]}")
    
    # Create a new DataFrame with the intervals
    result_df = pd.DataFrame({
        "interval": intervals,
        "deaths": processed_df["deaths"].iloc[:-1].values,
        "cens": processed_df["cens"].iloc[:-1].values
    })
    
    return result_df

def calculate_statistics(results_df):
    """
    Calculate additional statistics based on the survival analysis results
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame with survival analysis results
        
    Returns:
    --------
    Dictionary of additional statistics
    """
    stats = {}
    
    # Calculate median survival time (where survival = 0.5)
    for i in range(len(results_df) - 1):
        if results_df.loc[i, "surv"] >= 0.5 and results_df.loc[i+1, "surv"] < 0.5:
            # Linear interpolation
            interval1 = results_df.loc[i, "interval"].split("-")
            interval2 = results_df.loc[i+1, "interval"].split("-")
            
            t1 = float(interval1[1])
            t2 = float(interval2[1])
            
            s1 = results_df.loc[i, "surv"]
            s2 = results_df.loc[i+1, "surv"]
            
            # Interpolation: t = t1 + (0.5 - s1) * (t2 - t1) / (s2 - s1)
            median = t1 + (0.5 - s1) * (t2 - t1) / (s2 - s1)
            stats["median_survival"] = median
            break
    
    # If no median was found (all survival probabilities > 0.5 or < 0.5)
    if "median_survival" not in stats:
        if results_df["surv"].iloc[0] < 0.5:
            stats["median_survival"] = 0  # Everyone died before first interval
        else:
            stats["median_survival"] = float('inf')  # More than half survived the entire period
    
    # Calculate 1-year and 5-year survival rates
    for i, row in results_df.iterrows():
        interval = row["interval"].split("-")
        start, end = float(interval[0]), float(interval[1])
        
        # 1-year survival
        if start <= 1 and end > 1:
            # Interpolate
            if i > 0:
                prev_surv = results_df.loc[i-1, "surv"]
                curr_surv = row["surv"]
                t_ratio = (1 - start) / (end - start)
                stats["one_year_survival"] = prev_surv + t_ratio * (curr_surv - prev_surv)
            else:
                stats["one_year_survival"] = row["surv"]
        
        # 5-year survival
        if start <= 5 and end > 5:
            # Interpolate
            if i > 0:
                prev_surv = results_df.loc[i-1, "surv"]
                curr_surv = row["surv"]
                t_ratio = (5 - start) / (end - start)
                stats["five_year_survival"] = prev_surv + t_ratio * (curr_surv - prev_surv)
            else:
                stats["five_year_survival"] = row["surv"]
    
    # If specific time points weren't found in intervals
    if "one_year_survival" not in stats:
        # Find closest interval
        closest_idx = results_df["interval"].apply(
            lambda x: abs(1 - float(x.split("-")[1]))
        ).idxmin()
        stats["one_year_survival"] = results_df.loc[closest_idx, "surv"]
    
    if "five_year_survival" not in stats:
        # Find closest interval
        closest_idx = results_df["interval"].apply(
            lambda x: abs(5 - float(x.split("-")[1]))
        ).idxmin()
        stats["five_year_survival"] = results_df.loc[closest_idx, "surv"]
    
    return stats