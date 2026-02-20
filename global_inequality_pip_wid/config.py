"""
Configuration file for global inequality analysis project.

Centralizes all parameters, file paths, and settings used across the pipeline.
"""

import os

# =============================================================================
# PROJECT DIRECTORIES
# =============================================================================

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
MODIFIED_DIR = os.path.join(BASE_DIR, "modified")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Create directories if they don't exist
for directory in [INPUTS_DIR, MODIFIED_DIR, OUTPUTS_DIR]:
    os.makedirs(directory, exist_ok=True)


# =============================================================================
# DATA PARAMETERS
# =============================================================================

# Year to analyze (WID data is only available for 2023 currently)
TARGET_YEAR = 2023

# PIP data URL
PIP_URL = "https://catalog.ourworldindata.org/garden/wb/2025-10-13/thousand_bins_distribution/thousand_bins_distribution.feather?nocache"


# =============================================================================
# INPUT FILES (Raw data from WID)
# =============================================================================

# WID input files (created by Stata .do file)
WID_PPP_FILE = os.path.join(INPUTS_DIR, "WID_ppp.csv")
WID_PERCENTILES_FILE = os.path.join(INPUTS_DIR, "WID_percentiles.csv")
WID_POPULATION_FILE = os.path.join(INPUTS_DIR, "WID_aggregate_population.csv")
COUNTRY_MAPPING_FILE = os.path.join(INPUTS_DIR, "country_mapping.csv")

# Stata .do files (used by fetch_wid_data.py)
STATA_PPP_DO_FILE = os.path.join(INPUTS_DIR, "fetch_ppp.do")
STATA_POPULATION_DO_FILE = os.path.join(INPUTS_DIR, "fetch_population.do")


# =============================================================================
# INTERMEDIATE FILES (Processed data)
# =============================================================================

WID_WITH_PER_CAPITA_FILE = os.path.join(MODIFIED_DIR, "WID_percentiles_with_per_capita.csv")
HARMONIZED_FILE = os.path.join(MODIFIED_DIR, "pip_wid_harmonized.csv")


# =============================================================================
# OUTPUT FILES (Analysis results)
# =============================================================================

# Charts and visualizations
INEQUALITY_DECOMPOSITION_CHART = os.path.join(OUTPUTS_DIR, "inequality_decomposition.png")
MEAN_INCOME_PLOT = os.path.join(OUTPUTS_DIR, "pip_vs_wid_mean_income_interactive.html")
COUNTRY_MLD_PLOT = os.path.join(OUTPUTS_DIR, "pip_vs_wid_country_mlds.html")
COUNTERFACTUAL_CHART = os.path.join(OUTPUTS_DIR, "counterfactual_decomposition.png")
SHAPE_VS_MEAN_CHART = os.path.join(OUTPUTS_DIR, "shape_vs_mean_decomposition.png")


# =============================================================================
# STATA CONFIGURATION
# =============================================================================

# Path to Stata executable
STATA_PATH = "/Applications/Stata/StataSE.app/Contents/MacOS/stata-se"


# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# Counterfactual scenarios (for counterfactual_analysis.py)
COUNTERFACTUAL_SCENARIOS = [
    ("Mexico", "United States"),
    ("Brazil", "United States"),
    ("India", "Pakistan"),
]
