import streamlit as st

st.set_page_config(
    page_title="Predcitive maintenance",
)

st.write("# Predictive maintenance")

st.sidebar.success("Report pages")

st.markdown(""" 
data : https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
S. Matzka, Explainable Artificial Intelligence for Predictive Maintenance Applications,
2020 Third International Conference on Artificial Intelligence for Industries (AI4I), 
2020, pp. 69-74, doi: 10.1109/AI4I49448.2020.00023.""")

st.markdown(""" ## Columns
- **UID**: unique identifier ranging from 1 to 10000
- **Product ID**: consisting of a letter L, M, or H for low (50% of all products), medium (30%) and high (20%) as product quality variants and a variant-specific serial number
- **Type**: just the product type L, M or H from column 2
- **Air temperature [K]**: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
- **Process temperature [K]**: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K
- **Rotational speed [rpm]**: calculated from a power of 2860 W, overlaid with a normally distributed noise
- **Torque [Nm]**: torque values are normally distributed around 40 Nm with a SD = 10 Nm and no negative values
- **Tool wear [min]**: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process
- **Machine failure label**: indicates whether the machine has failed in this particular datapoint for any of the following failure modes are true

The machine failure consists of five independent failure modes:

- **Tool wear failure (TWF)**: The tool will be replaced or fail at a randomly selected tool wear time between 200 - 240 mins (120 times in our dataset). At this point in time, the tool is replaced 69 times, and fails 51 times (randomly assigned).
- **Heat dissipation failure (HDF)**: Heat dissipation causes a process failure if the difference between air and process temperature is below 8.6 K and the tool's rotational speed is below 1380 rpm. This is the case for 115 data points.
- **Power failure (PWF)**: The product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails, which is the case 95 times in our dataset.
- **Overstrain failure (OSF)**: If the product of tool wear and torque exceeds 11,000 minNm for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain. This is true for 98 data points.
- **Random failures (RNF)**: Each process has a 0.1% chance to fail regardless of its process parameters. This is the case for only 5 data points, less than expected for 10,000 datapoints in our dataset.

If at least one of the above failure modes is true, the process fails and the 'machine failure' label is set to 1. It is not transparent to the machine learning method which of the failure modes has caused the process to fail.
""")
