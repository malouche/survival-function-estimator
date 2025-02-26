import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from modules.data_processor import load_data, validate_data, preprocess_data
from modules.survival_estimator import estimate_survival_function
from modules.visualizer import plot_survival_curve, plot_hazard_function, plot_pdf
from modules.about import show_about
from modules.feedback import show_feedback

# Set page config
st.set_page_config(
    page_title="Survival Function Estimator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
    }
</style>
""", unsafe_allow_html=True)

# Add title
st.title("Survival Function Estimation from Summary Data")
st.markdown("Estimate survival metrics using interval-censored data")

# Sidebar 
with st.sidebar:
    st.header("Data Input")
    
    # Option to use sample data or upload own data
    data_option = st.radio(
        "Choose data source:",
        options=["Upload CSV", "Use Sample Data", "Manual Input"],
    )
    
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload your CSV file", 
            type=["csv"], 
            help="File should have columns: intEndpoints, deaths, cens"
        )
        
        if uploaded_file is not None:
            try:
                df = load_data(uploaded_file)
                if validate_data(df):
                    st.success("Data loaded successfully!")
                else:
                    st.error("Data validation failed. Please check your CSV format.")
                    df = None
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                df = None
        else:
            df = None
            
    elif data_option == "Use Sample Data":
        try:
            # Load sample data
            df = pd.DataFrame({
                "intEndpoints": list(range(17)),  # 0-16
                "deaths": [456, 226, 152, 171, 135, 125, 83, 74, 51, 42, 43, 34, 18, 9, 6, 0],
                "cens": [0, 39, 22, 23, 24, 107, 133, 102, 68, 64, 45, 53, 33, 27, 23, 30]
            })
            st.success("Sample data loaded!")
            
            # Display sample data
            with st.expander("View Sample Data"):
                st.dataframe(df)
                
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            df = None
            
    elif data_option == "Manual Input":
        st.write("Enter your data points:")
        
        # Initial sample size
        sample_size = st.number_input("Initial Sample Size", min_value=1, value=2418)
        
        # Create dynamic input for intervals
        num_intervals = st.number_input("Number of intervals", min_value=1, max_value=20, value=16)
        
        manual_data = {
            "intEndpoints": [0],
            "deaths": [],
            "cens": []
        }
        
        for i in range(num_intervals):
            col1, col2, col3 = st.columns(3)
            with col1:
                endpoint = st.number_input(f"Endpoint {i+1}", value=i+1, key=f"endpoint_{i}")
                manual_data["intEndpoints"].append(endpoint)
            with col2:
                deaths = st.number_input(f"Deaths in interval {i}", min_value=0, value=0, key=f"deaths_{i}")
                manual_data["deaths"].append(deaths)
            with col3:
                cens = st.number_input(f"Censored in interval {i}", min_value=0, value=0, key=f"cens_{i}")
                manual_data["cens"].append(cens)
        
        # Create DataFrame from manual input
        try:
            df = pd.DataFrame(manual_data)
            # Ensure intEndpoints are sorted
            df = df.sort_values("intEndpoints").reset_index(drop=True)
            
            # Display the manually entered data
            with st.expander("View Entered Data"):
                st.dataframe(df)
                
        except Exception as e:
            st.error(f"Error with manual data: {str(e)}")
            df = None

    # Add a divider
    st.markdown("---")
    
    # Additional options
    st.header("Analysis Options")
    
    # Option for confidence intervals
    show_ci = st.checkbox("Show confidence intervals", value=True)
    ci_level = st.slider("Confidence level (%)", min_value=80, max_value=99, value=95)

# Main panel - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Results", "Code Snippets", "About", "Feedback"])

with tab1:
    # Results tab
    if df is not None:
        st.header("Survival Analysis Results")
        
        # Preprocess data
        processed_data = preprocess_data(df)
        
        # Calculate survival metrics
        results = estimate_survival_function(processed_data, sample_size=sample_size if data_option == "Manual Input" else 2418)
        
        # Display results
        st.subheader("Detailed Calculations")
        st.dataframe(results)
        
        # Create columns for different plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Survival Function")
            fig_survival = plot_survival_curve(results, show_ci=show_ci, ci_level=ci_level)
            st.pyplot(fig_survival)
            
        with col2:
            st.subheader("Hazard Function")
            fig_hazard = plot_hazard_function(results)
            st.pyplot(fig_hazard)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Probability Density Function")
            fig_pdf = plot_pdf(results)
            st.pyplot(fig_pdf)
            
        with col2:
            st.subheader("Summary Statistics")
            
            # Calculate median survival time (where survival drops below 0.5)
            median_survival = None
            for i in range(len(results)-1):
                if results.loc[i, 'surv'] >= 0.5 and results.loc[i+1, 'surv'] < 0.5:
                    # Linear interpolation to get more precise median
                    y1, y2 = results.loc[i, 'surv'], results.loc[i+1, 'surv']
                    x1, x2 = results.loc[i, 'interval'].split('-')[1], results.loc[i+1, 'interval'].split('-')[1]
                    x1, x2 = float(x1), float(x2)
                    median_survival = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1)
                    break
            
            if median_survival:
                st.metric("Median Survival Time", f"{median_survival:.2f} time units")
                
            # Calculate restricted mean survival time (area under the curve)
            rmst = 0
            for i in range(len(results)-1):
                if i < len(results)-1:  # Skip the last interval
                    interval_start = float(results.loc[i, 'interval'].split('-')[0])
                    interval_end = float(results.loc[i, 'interval'].split('-')[1])
                    interval_width = interval_end - interval_start
                    avg_surv = (results.loc[i, 'surv'] + results.loc[i+1, 'surv']) / 2
                    rmst += interval_width * avg_surv
            
            st.metric("Restricted Mean Survival Time", f"{rmst:.2f} time units")
            
            # Calculate 1-year and 5-year survival rates
            one_year_survival = None
            five_year_survival = None
            
            for i, row in results.iterrows():
                interval = row['interval']
                if interval == "0-1":
                    one_year_survival = row['surv']
                if interval == "4-5" or interval == "5-6":  # Depending on how intervals are defined
                    five_year_survival = row['surv']
            
            if one_year_survival:
                st.metric("1-Year Survival Rate", f"{one_year_survival:.2%}")
            if five_year_survival:
                st.metric("5-Year Survival Rate", f"{five_year_survival:.2%}")
        
        # Export options
        st.subheader("Export Results")
        col1, col2, col3 = st.columns(3)
        
        # Function to download data as CSV
        def get_csv_download_link(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="survival_results.csv">Download CSV</a>'
            return href
        
        # Function to download plot as PNG
        def get_png_download_link(fig, filename):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
            buf.seek(0)
            b64 = base64.b64encode(buf.getvalue()).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download PNG</a>'
            return href
        
        # Function to download all plots as a single PDF
        def get_pdf_download_link(figs, filename="survival_plots.pdf"):
            buf = io.BytesIO()
            with PdfPages(buf) as pdf:
                for fig in figs:
                    fig.savefig(bbox_inches="tight")
                    pdf.savefig(fig)
            buf.seek(0)
            b64 = base64.b64encode(buf.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF</a>'
            return href
        
        with col1:
            st.markdown(get_csv_download_link(results), unsafe_allow_html=True)
            
        with col2:
            st.markdown(get_png_download_link(fig_survival, "survival_curve.png"), unsafe_allow_html=True)
            
        with col3:
            st.markdown(get_pdf_download_link([fig_survival, fig_hazard, fig_pdf]), unsafe_allow_html=True)
    
    else:
        # No data loaded yet
        st.info("Please upload or select data in the sidebar to view results.")
        st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=200)

with tab2:
    # Code snippets tab
    st.header("Code Snippets")
    st.write("You can use these code snippets to perform similar survival analysis in R or Python.")
    
    # Python code
    st.subheader("Python Code")
    python_code = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def life_table(intervals, deaths, censored, n_init):
    """
    Calculate life table estimates for survival data.
    
    Parameters:
    -----------
    intervals : list
        Time interval endpoints
    deaths : list
        Number of deaths in each interval
    censored : list
        Number of censored observations in each interval
    n_init : int
        Initial sample size
        
    Returns:
    --------
    DataFrame containing life table estimates
    """
    # Initialize results dataframe
    results = pd.DataFrame()
    
    # Create interval labels
    interval_labels = [f"{intervals[i]}-{intervals[i+1]}" for i in range(len(intervals)-1)]
    results['interval'] = interval_labels
    
    # Number of subjects at start
    n_subs = [n_init]
    for i in range(1, len(deaths)):
        n_subs.append(n_subs[i-1] - deaths[i-1] - censored[i-1])
    results['nsubs'] = n_subs[:-1]  # Remove the last element
    
    # Number lost to censoring
    results['nlost'] = censored[:-1]
    
    # Number at risk (adjusted for censoring)
    results['nrisk'] = [n - c/2 for n, c in zip(results['nsubs'], results['nlost'])]
    
    # Number of events (deaths)
    results['nevent'] = deaths[:-1]
    
    # Survival probability for each interval
    results['prob'] = [1 - d/r if r > 0 else 0 for d, r in zip(results['nevent'], results['nrisk'])]
    
    # Survival function
    results['surv'] = np.cumprod(np.insert(results['prob'], 0, 1.0))[1:]
    
    # Probability density function
    results['pdf'] = [s * (1 - p) for s, p in zip(results['surv'], results['prob'])]
    
    # Hazard function
    interval_widths = [intervals[i+1] - intervals[i] for i in range(len(intervals)-1)]
    results['hazard'] = [(1 - p) / w for p, w in zip(results['prob'], interval_widths)]
    
    # Standard errors
    results['se_surv'] = [s * np.sqrt(np.sum([d / (r * (r - d)) if r > d and r > 0 else 0 
                                           for d, r in zip(results['nevent'][:i+1], results['nrisk'][:i+1])])) 
                        for i, s in enumerate(results['surv'])]
    
    return results

# Example usage with the provided data
intervals = list(range(17))  # 0-16
deaths = [456, 226, 152, 171, 135, 125, 83, 74, 51, 42, 43, 34, 18, 9, 6, 0]
censored = [0, 39, 22, 23, 24, 107, 133, 102, 68, 64, 45, 53, 33, 27, 23, 30]
n_init = 2418

results = life_table(intervals, deaths, censored, n_init)
print(results)

# Plot survival curve
plt.figure(figsize=(10, 6))
plt.step(
    [float(i.split('-')[0]) for i in results['interval']], 
    results['surv'], 
    where='post', 
    label='Survival Function'
)
plt.fill_between(
    [float(i.split('-')[0]) for i in results['interval']], 
    results['surv'] - 1.96 * results['se_surv'],
    results['surv'] + 1.96 * results['se_surv'],
    alpha=0.2, step='post'
)
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title('Survival Function Estimate')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.legend()
plt.show()
'''
    st.code(python_code, language="python")
    
    # R code
    st.subheader("R Code")
    r_code = '''
library(survival)
library(KMsurv)
library(ggplot2)

# Define the data
intEndpts <- 0:16  # Interval endpoints
deaths <- c(456, 226, 152, 171, 135, 125, 83, 74, 51, 42, 43, 34, 18, 9, 6, 0)
cens <- c(0, 39, 22, 23, 24, 107, 133, 102, 68, 64, 45, 53, 33, 27, 23, 30)

# Use the lifetab function from KMsurv package
lt_result <- lifetab(tis = intEndpts, ninit = 2418, nlost = cens, nevent = deaths)
print(lt_result)

# Create a data frame for plotting
lt_df <- data.frame(
  time = intEndpts[-length(intEndpts)],
  surv = lt_result$surv,
  lower = pmax(0, lt_result$surv - 1.96 * lt_result$se.surv),
  upper = pmin(1, lt_result$surv + 1.96 * lt_result$se.surv)
)

# Plot with ggplot2
ggplot(lt_df, aes(x = time, y = surv)) +
  geom_step(direction = "hv", size = 1) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, stat = "stepribbon", direction = "hv") +
  labs(
    title = "Survival Function Estimate",
    x = "Time",
    y = "Survival Probability"
  ) +
  theme_minimal() +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90")
  ) +
  scale_y_continuous(limits = c(0, 1.05))

# Alternative approach using survfit with interval-censored data
# Create a custom interval-censored dataset
n <- sum(deaths) + sum(cens)  # Total sample size
event_times <- rep(0, n)
status <- rep(0, n)  # 0=censored, 1=event
interval_start <- rep(0, n)
interval_end <- rep(0, n)

idx <- 1
for (i in 1:(length(intEndpts)-1)) {
  # Add events for this interval
  if (deaths[i] > 0) {
    event_idx <- idx:(idx + deaths[i] - 1)
    status[event_idx] <- 1
    interval_start[event_idx] <- intEndpts[i]
    interval_end[event_idx] <- intEndpts[i+1]
    idx <- idx + deaths[i]
  }
  
  # Add censored for this interval
  if (cens[i] > 0) {
    cens_idx <- idx:(idx + cens[i] - 1)
    status[cens_idx] <- 0
    interval_start[cens_idx] <- intEndpts[i]
    interval_end[cens_idx] <- intEndpts[i+1]
    idx <- idx + cens[i]
  }
}

# Create a Surv object with interval censoring
surv_obj <- Surv(time = interval_start, time2 = interval_end, event = status, type = "interval")

# Fit the survival curve
fit <- survfit(surv_obj ~ 1)
plot(fit, xlab = "Time", ylab = "Survival Probability", main = "Survival Curve (interval-censored)")
'''
    st.code(r_code, language="r")

with tab3:
    # About tab
    show_about()

with tab4:
    # Feedback tab
    show_feedback()

# Footer
st.markdown("---")
st.markdown("Survival Function Estimator - Developed based on Dhafer Malouche's methodology.")