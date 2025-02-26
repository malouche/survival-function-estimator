import streamlit as st

def show_about():
    st.header("About")
    st.write("**Dhafer Malouche, Professor of Statistics**")
    st.write("**Department of Mathematics and Statistics**")
    st.write("**College of Arts and Sciences, Qatar University**")
    st.write("**Email:** [dhafer.malouche@qu.edu.qa](mailto:dhafer.malouche@qu.edu.qa)")
    st.markdown("**Website:** [dhafermalouche.net](http://dhafermalouche.net)")
    
    st.markdown("---")
    
    st.subheader("About This Application")
    st.write("""
    This Streamlit application implements the survival function estimation methodology 
    described by Professor Dhafer Malouche for estimating survival metrics when only 
    summary data is available.
    
    The application allows users to:
    
    1. Upload their own interval-censored survival data
    2. Use sample data provided with the application
    3. Manually input their own data
    
    The application then calculates and visualizes:
    
    - Survival function
    - Hazard function
    - Probability density function
    - Various summary statistics
    
    All calculations follow the methodology outlined in "Estimating Survival Metrics 
    Without Raw Data" from the STAT 442 course materials.
    """)
    
    st.subheader("Methodology")
    st.write("""
    The survival analysis implemented in this application follows these steps:
    
    1. Calculate the adjusted number at risk, accounting for censoring
    2. Estimate the conditional probability of dying in each interval
    3. Calculate the conditional probability of surviving each interval
    4. Compute the cumulative survival probability at each time point
    5. Estimate the hazard function and probability density function
    6. Calculate standard errors and confidence intervals
    
    For more details on the methodology, please refer to the course materials 
    or contact Professor Malouche directly.
    """)
