package edu.ucr.cs.cs167.project_section_23_4

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.RegressionEvaluator

object LinearRegression {

  def main(args: Array[String]): Unit = {
    if (args.length != 1) {
      println("Usage <input parquet file>")
      sys.exit(0)
    }
    val inputFile = args(0)
    val conf = new SparkConf()
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    println(s"Using Spark master '${conf.get("spark.master")}'")

    val spark = SparkSession
      .builder()
      .appName("Wildfire Prediction Model")
      .config(conf)
      .getOrCreate()

    //spark.conf.set("spark.sql.shuffle.partitions", "5") // Control partitioning

    val t1 = System.nanoTime()
    try {
      // load data
      val wildfireData: DataFrame = spark.read.parquet(inputFile)

      // process dates and group by County, year, month
      val dfWithDate = wildfireData.withColumn("date", to_date(col("acq_date"), "yyyy-MM-dd"))
      val dfWithYearMonth = dfWithDate
        .withColumn("year", year(col("date")))
        .withColumn("month", month(col("date")))
        .drop("date", "acq_date")

      // aggregate
      val aggregated = dfWithYearMonth.groupBy("County", "year", "month")
        .agg(
          sum("frp").alias("fire_intensity"),
          avg("ELEV_mean").alias("ELEV_mean"),
          avg("SLP_mean").alias("SLP_mean"),
          avg("EVT_mean").alias("EVT_mean"),
          avg("EVH_mean").alias("EVH_mean"),
          avg("CH_mean").alias("CH_mean"),
          avg("TEMP_ave").alias("TEMP_ave"),
          avg("TEMP_min").alias("TEMP_min"),
          avg("TEMP_max").alias("TEMP_max")
        )

      // split training/test split 80/20
      val Array(trainingData, testData) = aggregated.randomSplit(Array(0.8, 0.2))

      // define features
      val featureCols = Array(
        "ELEV_mean", "SLP_mean", "EVT_mean", "EVH_mean",
        "CH_mean", "TEMP_ave", "TEMP_min", "TEMP_max"
      )

      val assembler = new VectorAssembler()
        .setInputCols(featureCols)
        .setOutputCol("features")

      val scaler = new StandardScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
        .setWithMean(true)
        .setWithStd(true)

      val lr = new LinearRegression()
        .setFeaturesCol("scaledFeatures")
        .setLabelCol("fire_intensity")
        .setPredictionCol("prediction")

      // create then run pipeline
      val pipeline = new Pipeline()
        .setStages(Array(assembler, scaler, lr))

      val model: PipelineModel = pipeline.fit(trainingData)

      // make predictions on test data
      val predictions = model.transform(testData)

      val selectedColumns = featureCols ++ Array("fire_intensity", "prediction")
      predictions.select(selectedColumns.map(col): _*).show(10, truncate = false)

      // evaluate model by calculating root mean squre error
      val evaluator = new RegressionEvaluator()
        .setLabelCol("fire_intensity")
        .setPredictionCol("prediction")
        .setMetricName("rmse")

      val rmse = evaluator.evaluate(predictions)
      println(s"Root Mean Squared Error (RMSE): $rmse")

      val t2 = System.nanoTime()
      println(s"Total execution time: ${(t2 - t1) * 1e-9} seconds")

      // doesn't do well rn bc total dataset only has like 13 datapoints

    } finally {
      spark.stop()
    }
  }
}