# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ This File Contains Classes --------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# A class to handle data loading from BigQuery and retrieval in Spark
class DataToSpark():
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

# ------------------------------------------------------------------------------------------------------------------------    
print("The classes file is imported ☑")
