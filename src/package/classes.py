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
        
# A class A class for merging member data and transaction ________________________________________________________________
class ModelSelector:
    """Class for selecting the best model from a set of classifiers."""
    
    def __init__(self, train_df, test_df, assembler, indexer):
        """Initialize the ModelSelector with training and testing data, feature assembler, and label indexer."""
        
        # DataFrames for training and testing
        self.train_df = train_df
        self.test_df = test_df
        
        # Assembler and indexer for feature transformation
        self.assembler = assembler
        self.indexer = indexer
        
        # Define the models and their parameter grids for hyperparameter tuning
        self.models_and_params = {
            'Decision Tree': (
                DecisionTreeClassifier(),
                ParamGridBuilder().addGrid(DecisionTreeClassifier.maxDepth, [2, 5, 10, 20])
                                  .addGrid(DecisionTreeClassifier.maxBins, [20, 40, 80]).build()
            ),
            'Random Forest': (
                RandomForestClassifier(), 
                ParamGridBuilder().addGrid(RandomForestClassifier.numTrees, [10, 20])
                                  .addGrid(RandomForestClassifier.maxDepth, [2, 5, 10]).build()
            ),
            'Gradient-Boosted Tree': (
                GBTClassifier(), 
                ParamGridBuilder().addGrid(GBTClassifier.maxIter, [10, 20])
                                  .addGrid(GBTClassifier.maxDepth, [2, 5, 10]).build()
            ),
            'Linear Support Vector Machine': (
                LinearSVC(), 
                ParamGridBuilder().addGrid(LinearSVC.regParam, [0.1, 0.01])
                                  .addGrid(LinearSVC.maxIter, [100, 200]).build()
            )
        }
        
        # Initialize variables to store the best model and its metrics
        self.best_model = None
        self.best_model_name = None
        self.best_model_params = None
        self.best_f1_score = 0
        
        # Initialize DataFrame to store evaluation results
        self.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        self.schema = StructType([StructField(name, FloatType(), True) for name in self.columns])
        self.results = spark.createDataFrame([], self.schema)

    def fit(self):
        """Fit the models and select the best one based on F1 score."""
        
        # Loop through each model and its parameter grid
        for model_name, (model, paramGrid) in self.models_and_params.items():
            
            # Create a pipeline with feature transformation and model
            pipeline = Pipeline(stages=[self.assembler, self.indexer, model])
            
            # Perform cross-validation
            crossval = CrossValidator(estimator=pipeline,
                                      estimatorParamMaps=paramGrid,
                                      evaluator=MulticlassClassificationEvaluator(metricName="f1"),
                                      numFolds=5) 

            # Fit the model
            cvModel = crossval.fit(self.train_df)
            
            # Make predictions on the test set
            prediction = cvModel.transform(self.test_df)

            # Get evaluation metrics
            accuracy = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(prediction)
            precision = MulticlassClassificationEvaluator(metricName="weightedPrecision").evaluate(prediction)
            recall = MulticlassClassificationEvaluator(metricName="weightedRecall").evaluate(prediction)
            f1_score = MulticlassClassificationEvaluator(metricName="f1").evaluate(prediction)
            auc = BinaryClassificationEvaluator(metricName="areaUnderROC").evaluate(prediction)

            # Store the results
            result_row = (model_name, accuracy, precision, recall, f1_score, auc)
            self.results = self.results.union(spark.createDataFrame([result_row], ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']))

            # Update the best model if the current model has a higher F1 score
            if f1_score > self.best_f1_score:
                self.best_f1_score = f1_score
                self.best_model = cvModel.bestModel
                self.best_model_name = model_name
                self.best_model_params = cvModel.bestModel.extractParamMap()

    def display_results(self):
        """Display the evaluation results and details of the best model."""
        
        print(f"Best model: {self.best_model_name} with F1 score: {self.best_f1_score}")
        
        # Display the parameters of the best model
        for param, value in self.best_model_params.items():
            print(f"Parameter: {param.name}, Value: {value}")

        # Show the evaluation results
        self.results.show()
        
# ------------------------------------------------------------------------------------------------------------------------