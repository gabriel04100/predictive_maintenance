import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.transform import *
from src.visuals import summary_statistics, plot_correlation_matrix, check_missing_values, detect_outliers, plot_boxplot, countplot_percentage


st.set_page_config(page_title="EDA App")


# Cache the data loading and transformation step
@st.cache_data
def load_and_transform_data(data_path: str) -> pd.DataFrame:
    """
    Loads and applies transformations to the dataset. Caches the result for better performance.
    """
    df = pd.read_csv(data_path, index_col="UDI")
    df = transform_maintenance_data(df=df, speed_column="Rotational speed [rpm]",
                                    torque_column="Torque [Nm]", col1="Process temperature [K]",
                                    col2="Air temperature [K]", result_col="temp_diff [K]")
    return df


# Cache the correlation matrix plot with adjusted figure size
@st.cache_resource
def get_correlation_matrix_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Plots and caches the correlation matrix plot with a controlled size.
    """
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjust the size here
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.title("Correlation Matrix")
    return fig


# Cache the boxplot with adjusted figure size
@st.cache_resource
def get_boxplot(df: pd.DataFrame, column: str) -> plt.Figure:
    """
    Plots and caches the boxplot for a selected column with a controlled size.
    """
    fig, ax = plt.subplots(figsize=(5, 3))  # Adjust the size here
    sns.boxplot(data=df, x=column, ax=ax)
    plt.title(f"Boxplot of {column}")
    plt.grid()
    return fig


# Cache the countplot with adjusted figure size
@st.cache_resource
def get_countplot_percentage(df: pd.DataFrame, column: str, title: str) -> plt.Figure:
    """
    Plots and caches the countplot with percentages for a selected column with a controlled size.
    """
    fig, ax = plt.subplots(figsize=(5, 3))  # Adjust the size here
    sns.countplot(data=df, x=column, ax=ax)
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = '{:.1f}%'.format(100 * height / total)
        ax.text(p.get_x() + p.get_width() / 2., height + 0.5, percentage, ha="center")
    plt.title(title)
    plt.grid()
    return fig


# Function to plot histogram
@st.cache_resource
def get_histogram(df: pd.DataFrame, column: str) -> plt.Figure:
    """
    Plots a histogram for a selected column with controlled size.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[column], ax=ax)
    plt.title(f"Histogram of {column}")
    plt.grid()
    return fig


# Function to plot 2D scatter plot
@st.cache_resource
def get_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str) -> plt.Figure:
    """
    Plots a scatter plot for two selected numerical columns.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    plt.title(f"Scatter Plot: {x_col} vs {y_col}")
    plt.grid()
    return fig


def main():
    st.title("Exploratory Data Analysis (EDA) App")
    st.write("This page allows you to perform Exploratory Data Analysis on the dataset.")

    # Load and cache the transformed dataset
    data_path = "data/data.csv"
    try:
        df = load_and_transform_data(data_path)
        st.subheader("Dataset Overview")
        st.write(df.head())  # Display the first few rows of the dataset

        # Sidebar options for EDA
        st.sidebar.subheader("EDA Options")
        show_summary = st.sidebar.checkbox("Show Summary Statistics")
        show_corr_matrix = st.sidebar.checkbox("Show Correlation Matrix")
        show_missing_values = st.sidebar.checkbox("Check Missing Values")
        show_outliers = st.sidebar.checkbox("Detect Outliers")
        show_boxplot = st.sidebar.checkbox("Show Boxplot")
        show_countplot = st.sidebar.checkbox("Show Countplot with Percentages")
        show_histogram = st.sidebar.checkbox("Show Histogram")
        show_scatter_plot = st.sidebar.checkbox("Show 2D Scatter Plot")

        # Show Summary Statistics
        if show_summary:
            st.subheader("Summary Statistics")
            st.write(summary_statistics(df))

        # Show Correlation Matrix
        if show_corr_matrix:
            st.subheader("Correlation Matrix")
            st.write("The heatmap below shows the correlation between numerical variables.")
            corr_fig = get_correlation_matrix_plot(df.select_dtypes(['int','float']))
            st.pyplot(corr_fig)

        # Check Missing Values
        if show_missing_values:
            st.subheader("Missing Values")
            missing_values = check_missing_values(df)
            st.write(missing_values)

        # Detect Outliers
        if show_outliers:
            st.subheader("Detect Outliers")
            column = st.selectbox("Select column to detect outliers", df.select_dtypes(include='number').columns)
            threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0)
            outliers = detect_outliers(df, column, threshold)
            st.write(outliers)

        # Show Boxplot
        if show_boxplot:
            st.subheader("Boxplot")
            column = st.selectbox("Select column for boxplot", df.select_dtypes(include='number').columns)
            box_fig = get_boxplot(df, column)
            st.pyplot(box_fig)

        # Show Countplot with Percentages
        if show_countplot:
            st.subheader("Countplot with Percentages")
            column = st.selectbox("Select categorical column for countplot", ["Type"])
            count_fig = get_countplot_percentage(df, column, title=f"Repartition of {column}")
            st.pyplot(count_fig)

        # Show Histogram
        if show_histogram:
            st.subheader("Histogram")
            column = st.selectbox("Select numerical column for histogram", df.select_dtypes(include='number').columns)
            hist_fig = get_histogram(df, column)
            st.pyplot(hist_fig)

        # Show 2D Scatter Plot
        if show_scatter_plot:
            st.subheader("2D Scatter Plot")
            x_col = st.selectbox("Select X-axis column", df.select_dtypes(include='number').columns)
            y_col = st.selectbox("Select Y-axis column", df.select_dtypes(include='number').columns)
            scatter_fig = get_scatter_plot(df, x_col, y_col)
            st.pyplot(scatter_fig)

    except FileNotFoundError:
        st.error(f"File not found at {data_path}. Please check the file path and try again.")

if __name__ == "__main__":
    main()


