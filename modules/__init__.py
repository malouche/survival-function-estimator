# This file makes the modules directory a Python package
# It can be left empty or include imports to simplify accessing modules
from modules.data_processor import load_data, validate_data, preprocess_data, calculate_statistics
from modules.survival_estimator import estimate_survival_function
from modules.visualizer import plot_survival_curve, plot_hazard_function, plot_pdf, plot_cumulative_hazard
from modules.about import show_about
from modules.feedback import show_feedback

# Version information
__version__ = "0.1.0"