# ------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- This File Contains Functions -------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# Dependencies and Setup
from package.config import *

"""
func.py - Comprehensive Data Processing and Analysis Toolkit -------------------------------------------------------------

This file serves as a multifunctional utility module, designed to facilitate various data processing and analysis tasks. 
It leverages PySpark's DataFrame functionalities and integrates with Google Cloud Storage. Below is a brief overview of 
each function.

Functions ----------------------------------------------------------------------------------------------------------------
|- avg_calc:
  |- Purpose: Computes the average value of a specified DataFrame column.
  |- Features: Returns the average along with confidence intervals. User can specify a desired confidence level.
|- bucketized:
  |- Purpose: Segments a DataFrame column into distinct buckets or bins.
  |- Features: Allows manual boundary specification. Supports automatic boundary generation based on start, end, and step 
  values.
|- noise_member:
  |- Purpose: Analyzes noise in a DataFrame focusing on a specified "msno" member.
  |- Features: Flags columns with NULL values. Identifies columns with multiple distinct values.
|- noise_finder:
  |- Purpose: Isolates and removes noisy data in a DataFrame.
  |- Features: Provides information on duplicate rows. Returns "msno" keys used for noise identification.
|- aggregate_user_activity:
  |- Purpose: Aggregates transaction and user activity data into a single DataFrame.
  |- Features: Joins DataFrames on "msno" field. Computes aggregated metrics for each "msno", like activity count, sum of 
  various categories, and total time.
|- save_model_data:
  |- Purpose: Exports a DataFrame to Google Cloud Storage.
  |- Features: Saves in Parquet format. Requires GCS bucket path and filename. Supports optional custom credential path for
  GCP. Each function has been crafted with error-handling capabilities, providing a robust user experience.
--------------------------------------------------------------------------------------------------------------------------
"""

# Average and Boundry Calculation ________________________________________________________________________________________
def avg_calc(df: DataFrame, column: str, confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculates the average of a column in a DataFrame along with a confidence interval.

    Args:
        df (DataFrame): A DataFrame containing the target column data.
        column (str): Target column for calculation.
        confidence_level (float, optional): The desired confidence level as a decimal. Defaults to 0.95.

    Returns:
        tuple: A tuple containing the average value, lower bound of the confidence interval,
               and upper bound of the confidence interval.
    """
    try:
        # Calculate the average of the target column
        average_value = df.agg(avg(column)).collect()[0][0]

        # Calculate the standard deviation of the target column
        stddev_value = df.selectExpr(f"stddev({column})").collect()[0][0]

        # Define the sample size
        sample_size = df.count()

        # Calculate the Z-score for the given confidence level
        z_score = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }.get(confidence_level, 1.96)  # default to 1.96 if the level is not recognized

        # Calculate the margin of error
        margin_of_error = z_score * (stddev_value / math.sqrt(sample_size))

        # Calculate the lower and upper bounds of the confidence interval
        lower_bound = average_value - margin_of_error
        upper_bound = average_value + margin_of_error

        return average_value, lower_bound, upper_bound

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None
    
# Bucketizer _____________________________________________________________________________________________________________
def bucketized(df: DataFrame, input_column: str, output_column: str, boundaries: Union[List[int], List[float]] = None) -> DataFrame:
    """
    Bucketizes a DataFrame based on specified boundaries.

    Args:
        df (DataFrame): The input DataFrame.
        input_column (str): The column to be bucketized.
        output_column (str): The column to store bucketized values.
        boundaries (list, optional): List of boundary values for bucketization, 
                                     or list of [start, end, step] for auto-generated boundaries.
                                     Default is None.

    Returns:
        DataFrame: The bucketized DataFrame.
    """
    try:
        # Convert the specified column to double data type
        df = df.withColumn(input_column, col(input_column).cast("double"))

        # Generate the boundaries manually if [start, end, step] is provided
        if boundaries is not None and len(boundaries) == 3:
            auto_boundaries = list(range(boundaries[0], boundaries[1] + boundaries[2], boundaries[2]))
            final_boundaries = [-float("inf")] + auto_boundaries + [float("inf")]
        elif boundaries is not None:
            final_boundaries = [-float("inf")] + boundaries + [float("inf")]
        else:
            raise ValueError("Invalid boundaries parameter.")

        # Create a Bucketizer transformer
        bucketizer = Bucketizer(
            splits=final_boundaries,
            inputCol=input_column,
            outputCol=output_column
        )

        # Transform the DataFrame using the Bucketizer
        bucketized_df = bucketizer.transform(df)
        return bucketized_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Noise Observation ______________________________________________________________________________________________________
def noise_member(df: DataFrame, member: str) -> None:
    """
    Analyzes noise in a DataFrame based on a specific "msno" member.

    Args:
        df (DataFrame): The input DataFrame to analyze.
        member (str): The "msno" member value to filter by.

    Returns:
        None: The function prints warning messages and displays a sample DataFrame.
    """

    try:
        # Filter DataFrame based on "msno" value
        noise_df = df.filter(col("msno") == member).cache()

        # Define aggregation expressions in a list comprehension
        agg_exprs = [
            func for col_name in noise_df.columns
            for func in [
                sum(when(col(col_name) == "NAN", 1).otherwise(0)).alias(f"{col_name}_null_count"),
                countDistinct(col_name).alias(f"{col_name}_distinct_count")
            ]
        ]

        # Perform aggregation in a single pass
        stats = noise_df.agg(*agg_exprs).collect()[0].asDict()

        # Display column statistics
        for col_name in noise_df.columns:
            null_count = stats.get(f"{col_name}_null_count")
            distinct_count = stats.get(f"{col_name}_distinct_count")

            if null_count is None or distinct_count is None:
                print(f"⚠️ Warning: Skipping {col_name} due to null stats")
                continue

            if null_count > 0:
                print(f"⚠️ Warning: Column [\033[1m{col_name}\033[0m] has NULL values")

            if distinct_count > 1:
                print(f"⚠️ Warning: Column [\033[1m{col_name}\033[0m] has {distinct_count} distinct values")

        # Show the filtered DataFrame
        noise_df.show(5)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Optional: Unpersist the cached DataFrame to free up memory
        noise_df.unpersist()

# Noise Finder____________________________________________________________________________________________________________
def noise_finder(df: DataFrame, param: list = None) -> (DataFrame, DataFrame, DataFrame):
    """
    Finds and filters noise in a DataFrame based on specified columns.

    Args:
        df (DataFrame): The input DataFrame.
        param (list, optional): List of column names for grouping and filtering. Default is None.

    Returns:
        DataFrame: Filtered DataFrame with noise removed.
        DataFrame: DataFrame containing information about duplicate rows.
        DataFrame: DataFrame containing "msno" keys used for noise identification.
    """
    if param is None:
        param = []

    try:
        # Group and count duplicate rows based on specified columns
        duplicate_row_info = df.groupBy(param).count().orderBy(desc("count"))
        
        # Filter rows with count > 1 (i.e., duplicate rows)
        duplicate_row_info = duplicate_row_info.filter(col("count") > 1)

        # Select the "msno" keys from the DataFrame containing duplicate rows
        duplicate_msno_keys = duplicate_row_info.select("msno")

        # Join original DataFrame with the selected "msno" keys to filter noise
        noise_filtered_df = df.join(duplicate_msno_keys, on=["msno"], how="inner")

        # Print DataFrame shape information
        print(f"DataFrame Shape:\n |-- Number of Records: {noise_filtered_df.count()}\n |-- Number of Unique Cases: {duplicate_row_info.count()}")

        return noise_filtered_df, duplicate_row_info, duplicate_msno_keys

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

# User Activity __________________________________________________________________________________________________________
def aggregate_user_activity(transaction_df: DataFrame, user_activity_df: DataFrame, member_df: DataFrame) -> DataFrame:
    """
    Joins transaction and user activity DataFrames, aggregates metrics, and displays information about the resulting DataFrame.

    Parameters:
    - transaction_df: DataFrame containing transaction data with column "msno".
    - user_activity_df: DataFrame containing user activity data with column "msno".
    - member_df: DataFrame containing member data with column "msno".

    Returns:
    - DataFrame: Aggregated DataFrame containing metrics for each "msno".
    """

    # Select only the "msno" column from the transaction DataFrame
    trans_keys = transaction_df.select("msno")
    
    # Inner join the user activity DataFrame with the transaction DataFrame on the "msno" column
    user_merge_df = user_activity_df.join(trans_keys, on=["msno"], how="inner")
    
    # Aggregate metrics for each "msno" in the joined DataFrame
    user_sum_df = user_merge_df.groupBy("msno").agg(
        format_number(count('date'), 0).alias('activity_count'),
        sum("num_25").alias("sum_num_25"),
        sum("num_50").alias("sum_num_50"),
        sum("num_75").alias("sum_num_75"),
        sum("num_985").alias("sum_num_985"),
        sum("num_100").alias("sum_num_100"),
        sum("num_unq").alias("sum_num_unq"),
        sum("total_secs").alias("total_secs")
    )
    
    # Inner join the aggregated DataFrame with the original transaction DataFrame and member DataFrame
    model_df = transaction_df.join(user_sum_df, on=["msno"], how="inner")
    member_df_final = member_df.join(model_df, on=["msno"], how="inner")
    
    # Display DataFrame information ( from DataFrameInfoDisplay class)
    member_df_final_info = DataFrameInfoDisplay(member_df_final)
    member_df_final_info.display_info(show_schema=True, show_data=True)
    
    return member_df_final

# Model Data Export ______________________________________________________________________________________________________
def save_model_data(df: DataFrame, gcs_bucket: str, file_name: str, credential_path: str = './customer-churn-391917-150cab3a2647.json') -> None:
    """
    Save a DataFrame to Google Cloud Storage in Parquet format.

    Parameters:
    - df: DataFrame to be saved.
    - gcs_bucket: Google Cloud Storage bucket path.
    - file_name: File name under which DataFrame should be saved.
    - credential_path: Path to the GCP credential JSON file. Default is './customer-churn-391917-150cab3a2647.json'

    Returns:
    None
    """
    try:
        # Set the environment variable for credentials
        original_credential_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

        # Specify GCS path
        gcs_path = f"{gcs_bucket}/{file_name}"

        # Save DataFrame to GCS
        df.write.mode('overwrite').parquet(gcs_path)

        # Restore original credentials environment variable
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = original_credential_path

    except Exception as e:
        print(f"An error occurred: {e}")
        # Optionally, restore original credentials environment variable in case of error
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = original_credential_path

# ------------------------------------------------------------------------------------------------------------------------
