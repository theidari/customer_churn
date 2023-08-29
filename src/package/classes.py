# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ This File Contains Classes --------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
from package.config import *

# A class to handle data loading from BigQuery and retrieval in Spark ____________________________________________________
class DataToSpark:    
    def __init__(self, project_id, dataset_id):
        """
        A class for loading BigQuery tables into Spark DataFrames.
        
        This class provides methods to load specified tables from a BigQuery dataset
        into Spark DataFrames and store them in a dictionary for easy retrieval.
        
        Args:
            project_id (str): The ID of the Google Cloud project.
            dataset_id (str): The ID of the BigQuery dataset.
        """
        # Initialize the class with the specified project and dataset identifiers
        self.project_id = project_id
        self.dataset_id = dataset_id

    def load_tables(self, tables=[]):
        """
        Load specified tables into Spark DataFrames and store them in a dictionary.
        
        This method loads specified tables from the configured BigQuery dataset into
        Spark DataFrames and stores them in a dictionary with table names as keys.
        
        Args:
            tables (list): A list of table names to be loaded.
        """
        self.df_tables = {}  # Dictionary to store DataFrames
        for i, table in enumerate(tables):
            # Load the table into a DataFrame
            df_table = spark.read.format("bigquery").option("table", f"{self.project_id}:{self.dataset_id}.{table}").load()
            # Store the DataFrame in the dictionary
            self.df_tables[table] = df_table

    def get_tables(self):
        """
        Retrieve the stored Spark DataFrames.
        
        This method returns the dictionary containing the stored Spark DataFrames.
        
        Returns:
            dict: A dictionary containing table names as keys and corresponding Spark DataFrames as values.
        """
        return self.df_tables

# A class to display dataframe shape _____________________________________________________________________________________
class DataFrameInfoDisplay:
    """
    A class for displaying information about a DataFrame including shape, schema, and content.

    Args:
        dataframe (DataFrame): The DataFrame to be displayed.
    """

    def __init__(self, dataframe):
        """
        Initialize the DataFrameInfoDisplay class.

        Args:
            dataframe (DataFrame): The DataFrame to be displayed.
        """
        self.dataframe = dataframe

    def display_info(self, show_schema=True, show_data=True, num_rows=5):
        """
        Display information about the DataFrame including shape, schema, and content.

        Args:
            show_schema (bool): Whether to print the schema. Default is True.
            show_data (bool): Whether to show the contents of the DataFrame. Default is True.
            num_rows (int): Number of rows to display. Default is 5. Set to None to display all rows.
        """
        # Print the shape information of the DataFrame
        print(f"shape\n |-- rows: {self.dataframe.count()}\n |-- columns: {len(self.dataframe.columns)}")

        # Optionally print the schema (column names and their data types) of the DataFrame
        if show_schema:
            self.dataframe.printSchema()

        # Optionally show the contents of the DataFrame
        if show_data:
            if num_rows is None:
                self.dataframe.show(truncate=False)
            else:
                self.dataframe.show(num_rows, truncate=False)
                
# A class A class for merging member data and transaction ________________________________________________________________
class TransactionMerger:
    """
    A class for merging member data and transaction data using PySpark.

    Args:
        target_df (DataFrame): The DataFrame containing member data.
        transactions_df (DataFrame): The DataFrame containing transaction data.
    """

    def __init__(self, target_df, transactions_df):
        """
        Initialize the TransactionMerger class.

        Args:
            target_df (DataFrame): The DataFrame containing member data.
            transactions_df (DataFrame): The DataFrame containing transaction data.
        """
        self.member_model_df = target_df
        self.transactions_df = transactions_df
        self.tables = {"transactions": transactions_df}
    
    def merge_transactions(self):
        """
        Merge member data with transaction data.

        Returns:
            DataFrame: A DataFrame containing merged data.
        """
        # Perform the merge operation for members' output dataset and Transaction data
        transactions_merge_df = self.member_model_df.join(self.tables["transactions"], on=['msno'], how='left')

        # Define the window specification for ranking transactions within each member group
        window_spec = Window.partitionBy("msno").orderBy(col("transaction_date").desc())

        # Rank transactions within each member group
        ranked_df = transactions_merge_df.withColumn("rank", rank().over(window_spec))

        # Select rows with rank 1 (last transaction for each member)
        last_transaction_df = ranked_df.filter(col("rank") == 1).drop("rank")

        # Select rows with rank 2 (transaction before the last for each member)
        transaction_before_last_df = ranked_df.filter(col("rank") == 2).drop("rank")

        # Select specific columns and rename one of them
        selected_columns_df = transaction_before_last_df.select(
            col("msno"),
            col("membership_expire_date").alias("exp_last")
        )

        # Merge members' output dataset with transaction data
        new_tran_df = last_transaction_df.join(selected_columns_df, on=['msno'], how='left')
        
        # Convert date columns to correct date format
        new_tran_df = new_tran_df.withColumn("transaction_date",
                       when(col("transaction_date").isNotNull(),
                            to_date(col("transaction_date").cast("string"), "yyyyMMdd")
                           ).otherwise("NAN"))
        new_tran_df = new_tran_df.withColumn("membership_expire_date",
                       when(col("membership_expire_date").isNotNull(),
                            to_date(col("membership_expire_date").cast("string"), "yyyyMMdd")
                           ).otherwise("NAN"))
        new_tran_df = new_tran_df.withColumn("exp_last",
                       when(col("exp_last").isNotNull(),
                            to_date(col("exp_last").cast("string"), "yyyyMMdd")
                           ).otherwise("NAN"))
        
        return new_tran_df

    def display_merged_info(self, new_tran_df, show_schema=True, show_data=True, num_rows=5):
        """
        Display information about the merged DataFrame using the DataFrameInfoDisplay class.

        Args:
            new_tran_df (DataFrame): The merged DataFrame.
            show_schema (bool): Whether to print the schema. Default is True.
            show_data (bool): Whether to show the contents of the DataFrame. Default is True.
            num_rows (int): Number of rows to display. Default is 5. Set to None to display all rows.
        """
        info_display = DataFrameInfoDisplay(new_tran_df)
        info_display.display_info(show_schema, show_data, num_rows)

# ------------------------------------------------------------------------------------------------------------------------
