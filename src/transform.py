import numpy as np
import pandas as pd


def calculate_mechanical_power(df: pd.DataFrame, speed_column: str, torque_column: str) -> pd.Series:
    """
    Calculates the mechanical power in watts for each entry in the DataFrame.

    Mechanical power is given by the formula: P = Torque * Rotational Speed (in rad/s).
    The rotational speed must be converted from revolutions per minute (rpm) to radians per second.

    Args:
    df (pd.DataFrame): The DataFrame containing the data.
    speed_column (str): The name of the column in the DataFrame representing the rotational speed in rpm.
    torque_column (str): The name of the column in the DataFrame representing the torque in Nm.

    Returns:
    pd.Series: A Pandas Series containing the mechanical power in watts for each row in the DataFrame.
    """
    # Convert rotational speed from rpm to rad/s
    rotational_speed_rad_s = df[speed_column] * (2 * np.pi / 60)
    
    # Calculate mechanical power
    mechanical_power = df[torque_column] * rotational_speed_rad_s
    
    return mechanical_power
