import streamlit as st
import pandas as pd
from src.ml import load_model_and_split, prepare_data
import os

# Set page title and layout
st.set_page_config(page_title="Predictive Maintenance")

st.write("# Predictive Maintenance")

st.sidebar.success("Navigate to the report pages")

# Display metadata about the dataset and project
st.markdown("""
**Milling Machine Maintenance**
Dataset source: [Kaggle - Predictive Maintenance Dataset AI4I 2020](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)

This project focuses on detecting and predicting machine failures based on various factors.

## Features
- **UID**: unique identifier (1-10000)
- **Product ID**: Letter L, M, or H for product quality variants
- **Type**: Product type based on quality (L, M, or H)
- **Air temperature [K]**
- **Process temperature [K]**
- **Rotational speed [rpm]**
- **Torque [Nm]**
- **Tool wear [min]**
- **Machine failure label**: Indicator for machine failure

### Failure Modes
- **TWF**: Tool Wear Failure
- **HDF**: Heat Dissipation Failure
- **PWF**: Power Failure
- **OSF**: Overstrain Failure
- **RNF**: Random Failures
""")



