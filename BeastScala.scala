package edu.ucr.cs.cs167.project_section_23_4
import edu.ucr.cs.bdlab.beast.geolite.{Feature, IFeature}
import org.apache.spark.SparkConf
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._

object BeastScala {
  def main(args: Array[String]): Unit = {
    // Initialize Spark configuration
    val conf = new SparkConf().setAppName("Beast Example")
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")  // use local mode if no master is specified

    val spark = SparkSession.builder().config(conf).getOrCreate()
    SparkSQLRegistration.registerUDT
    SparkSQLRegistration.registerUDF(spark)

    // The first argument is the operation type.
    val operation: String = args(0)
    try {
      val t1 = System.nanoTime()
      operation match {
        // ... (other operations like count-by-county, convert, etc.)

        // New operation for wildfire aggregation based on county GEOID.
        case "wildfire-aggregate" =>
          // Expected arguments:
          // args(1): Path to wildfire Parquet file (e.g., "WildfireDB.parquet")
          // args(2): County name (e.g., "Riverside")
          if (args.length < 3) {
            System.err.println("Usage: wildfire-aggregate <wildfire_parquet_file> <county_name>")
            System.exit(1)
          }
          val wildfireInputFile = args(1)
          val countyName = args(2)

          // --- Step 1: Load county dataset (shapefile) and get GEOID ---
          val countiesDF = spark.read.format("shapefile").load("tl_2018_us_county.zip")
          countiesDF.createOrReplaceTempView("counties")
          val countyDF = spark.sql(
            s"""
               |SELECT * FROM counties
               |WHERE STATEFP = '06' AND NAME = '$countyName'
               |""".stripMargin)
          if (countyDF.head(1).isEmpty) {
            println(s"No county found with name '$countyName' in California.")
            System.exit(1)
          }
          val geoid = countyDF.head().getAs[String]("GEOID")
          println(s"Selected county '$countyName' with GEOID: $geoid")

          // --- Step 2: Load wildfire dataset ---
          // Note: Use the correct column name for the county in your wildfire data.
          // Based on your error message, it seems the column is named "County" (capitalized).
          val wildfireDF = spark.read.parquet(wildfireInputFile)
          wildfireDF.createOrReplaceTempView("wildfires")
          val filteredWildfiresDF = spark.sql(
            s"""
               |SELECT * FROM wildfires
               |WHERE County = '$geoid'
               |""".stripMargin)
          if (filteredWildfiresDF.head(1).isEmpty) {
            println(s"No wildfire records found for county GEOID $geoid")
            System.exit(0)
          }

          // --- Step 3: Extract year and month from acq_date and create a combined column ---
          // Assuming acq_date is in the format "yyyy-MM-dd"
          val wildfireWithDateDF = filteredWildfiresDF
            .withColumn("year", substring(col("acq_date"), 1, 4))
            .withColumn("month", substring(col("acq_date"), 6, 2))
            .withColumn("year_month", concat_ws("-", col("year"), col("month")))

          // Optionally, register the updated DataFrame as a view
          wildfireWithDateDF.createOrReplaceTempView("county_wildfires")

          // --- Step 4: Group by year_month and compute the total fire intensity (sum of frp) ---
          val resultDF = spark.sql(
            """
              |SELECT year_month, SUM(frp) AS fire_intensity
              |FROM county_wildfires
              |GROUP BY year_month
              |ORDER BY year_month
              |""".stripMargin)

          // --- Step 5: Write the result to a CSV file ---
          val outputFilename = s"wildfires${countyName}.csv"
          resultDF.coalesce(1)
            .write
            .option("header", "true")
            .mode("overwrite")
            .csv(outputFilename)
          println(s"Output saved to $outputFilename")

        case _ =>
          Console.err.println(s"Invalid operation '$operation'")
      }
      val t2 = System.nanoTime()
      println(s"Operation '$operation' took ${(t2 - t1) * 1E-9} seconds")
    } finally {
      spark.stop()
    }
  }
}