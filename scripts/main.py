from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SQLContext
import pyspark as ps
import numpy as np
import pipeline as p

if __name__ == '__main__':

    spark = (
            ps.sql.SparkSession.builder
            .master("local[4]")
            .appName("jampoq")
            .getOrCreate()
            )

    sc = spark.sparkContext

    # Read in data
    cell_accessories = 's3a://capstone-g65ds/data/reviews_Cell_Phones_and_Accessories_5.json'
    toys_games = 's3a://capstone-g65ds/data/reviews_Toys_and_Games_5.json'

    df = spark.read.json(toys_games)

    # Instantiate pipeline
    pipeline = p.Pipeline(df,spark)

    # Get filtered data with helpful label at threshold.
    pipeline.get_data(10,'.85')

    # Add first and second layer features.
    pipeline.add_first_layer_features()
    pipeline.add_sec_layer_features()

    pipeline.df.show(5)
