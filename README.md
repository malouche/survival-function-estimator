# Survival Function Estimator

A Streamlit application for estimating survival metrics from interval-censored summary data, based on the methodology described by Professor Dhafer Malouche.

## Features

- Upload CSV files with interval-censored survival data
- Use sample data or enter data manually
- Visualize survival function, hazard function, and probability density function
- Calculate key survival statistics (median survival time, 1-year and 5-year survival rates)
- Export results as CSV or plots as PNG/PDF
- Code snippets for performing similar analyses in Python and R

## Installation

### Prerequisites

- Python 3.8 or higher

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/malouche/survival-function-estimator.git
   cd survival-function-estimator
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Use the application:
   - Select data source (upload CSV, use sample data, or manual input) in the sidebar
   - View results in the main panel
   - Export results or plots as needed

## Data Format

The application expects data in the following format:

### CSV Format
The input CSV file should contain three columns:
- `intEndpoints`: Time interval endpoints
- `deaths`: Number of deaths in each interval
- `cens`: Number of censored observations in each interval

Example:
```
intEndpoints,deaths,cens
0,456,0
1,226,39
2,152,22
...
```

### Manual Input
When using manual input:
1. Specify the initial sample size
2. Enter the number of intervals
3. For each interval, provide:
   - Endpoint time
   - Number of deaths
   - Number of censored observations

## Methodology

The application follows the life table method for interval-censored data as described in "Estimating Survival Metrics Without Raw Data":

1. Calculate adjusted number at risk, accounting for censoring
2. Estimate conditional probabilities of dying and surviving
3. Compute cumulative survival probabilities
4. Calculate hazard function and probability density function
5. Estimate standard errors and confidence intervals

## Deployment on Streamlit Cloud

This application can be deployed using Streamlit Cloud:

1. Fork or clone this repository to your GitHub account
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository to Streamlit Cloud
4. Specify the main file path as "app.py"
5. Click "Deploy"

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Professor Dhafer Malouche for the methodology
- Streamlit for the framework
- STAT 442 course materials for the theoretical foundations