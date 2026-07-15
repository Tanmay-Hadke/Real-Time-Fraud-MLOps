import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_timestamp

def init_spark():
    return SparkSession.builder \
        .appName("FraudPipelineCompute") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

def compute_streaming_features(spark_session, raw_transaction_dict):
    """
    Transforms raw streaming logs into calibrated model-ready features.
    """
    # Create Spark Dataframe out of isolated streaming transaction payload
    df = spark_session.read.json(spark_session.sparkContext.parallelize([raw_transaction_dict]))
    
    # Feature Engineering Logic: Calculate deviation from baseline operational averages
    # In production EMR/Databricks, this joins over sliding window states
    processed_df = df.withColumn("spend_deviation", col("Amount") - lit(88.0))
    
    return processed_df.collect()[0].asDict()

if __name__ == "__main__":
    spark = init_spark()
    mock_payload = {"user_id": "USER_12345", "Amount": 250.50, "Time": 120.0}
    output = compute_streaming_features(spark, mock_payload)
    print(f"Spark Engineered Live Features Matrix:\n{output}")