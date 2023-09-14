# ------------------------------------------------------------------------------------------------------------------------
# ----------------------------------- All libraries and variables are defined in this file -------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# Standard Library Imports
import json
import math
import os

# Other Libraries Imports
import numpy as np
import pandas as pd
import inspect
from scipy.stats import norm
import plotly.graph_objs as go
from google.cloud import storage
from google.oauth2 import service_account
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# PySpark Imports
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (DecisionTreeClassifier, GBTClassifier,
                                      LinearSVC, LogisticRegression,
                                      RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import Bucketizer, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession, Window, functions, Row
from pyspark.sql.functions import (asc, avg, col, count, countDistinct, date_format,
                                   dayofmonth, desc, format_number, month, rank,
                                   round, sum, to_date, when, year, udf)
from pyspark.sql.types import FloatType, StructField, StructType
from sklearn.metrics import accuracy_score, roc_auc_score

# Typing Imports
from typing import List, Tuple, Union

# Initialize Spark Session
spark = SparkSession.builder.appName("kkbox_churn").getOrCreate()

