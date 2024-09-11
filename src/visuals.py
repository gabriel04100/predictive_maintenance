import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """ Provides summary statistics for numerical columns in the DataFrame. """
    return df.describe()


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """ Checks for missing values in the DataFrame. """
    return df.isnull().sum()


def countplot_percentage(df: pd.DataFrame, x_col:str="" ,title:str="", show_percentage:bool=True,figsize:Tuple=(10,10)) -> None:
    """
    shows countplot and percentage of categories
    """
    total = len(df)
    fig=plt.figure(figsize=figsize)
    ax = sns.countplot(data=df, x=x_col)
    plt.title(title)

    if show_percentage:
        for p in ax.patches:
            height = p.get_height()
            percentage = '{:.1f}%'.format(100 * height / total)
            ax.text(p.get_x() + p.get_width() / 2., height + 0.5, percentage, ha="center")

    plt.grid()
    return fig


def histplot(df: pd.DataFrame, x_col:str="", title:str="", figsize:Tuple=(10,10)):
    """
    shows histogram for continous variable
    """
    fig=plt.figure(figsize=figsize)
    sns.histplot(data=df, x=x_col)
    plt.grid()
    return fig


def detect_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Detects potential outliers in a specific column using the z-score method.
    """
    mean = df[column].mean()
    std = df[column].std()
    z_scores = (df[column] - mean) / std
    return df[np.abs(z_scores) > threshold]


def plot_boxplot(df: pd.DataFrame, column: str, figsize:Tuple=(10,10)) -> None:
    """
    Plots a boxplot for a given column to show the distribution and detect potential outliers.
    """
    fig=plt.figure(figsize=figsize)
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.grid()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, figsize:Tuple=(10,10)) -> None:
    """
    Plots a heatmap of the correlation matrix to show relationships between numerical variables.
    """
    fig=plt.figure(figsize=figsize)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f')
    plt.title("Correlation Matrix Heatmap")
    return fig
