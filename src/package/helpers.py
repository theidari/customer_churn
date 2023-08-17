# ------------------------------------------------------------------------------------------------------------------------
# ----------------------------- All libraries, variables and functions are defined in this file --------------------------
# ------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from pyspark.sql.functions import col, when, to_date
from pyspark.ml.feature import Bucketizer



# Calculation Functions --------------------------------------------------------------------------------------------------
# Average and Boundry Calculation ________________________________________________________________________________________
def avg_calc(df, column):
    """
    Calculates the average column data along with a confidence interval.
    
    This function calculates the average column data from a DataFrame along with a confidence interval
    using a specified confidence level. It uses the provided DataFrame to compute the necessary statistics.
    
    Args:
        df (DataFrame): A DataFrame containing the column data values.
        column: target column for calculation.
        
    Returns:
        tuple: A tuple containing the average churn percentage, lower bound of the confidence interval,
               and upper bound of the confidence interval.
    """
    # Calculate the average churn percentage
    average_value = df.agg(avg("churn_percentage")).collect()[0][0]

    # Calculate the standard deviation of the churn percentage
    stddev_value = df.selectExpr("stddev(churn_percentage)").collect()[0][0]

    # Define the sample size
    sample_size = df.count()

    # Define the desired confidence level (e.g., 95%)
    confidence_level = 0.95

    # Calculate the Z-score for the given confidence level
    z_score = 1.96  # For a 95% confidence level

    # Calculate the margin of error
    margin_of_error = z_score * (stddev_value / math.sqrt(sample_size))

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = average_value - margin_of_error
    upper_bound = average_value + margin_of_error

    return average_value, lower_bound, upper_bound
# ------------------------------------------------------------------------------------------------------------------------    
print("The helpers file is imported ☑")
