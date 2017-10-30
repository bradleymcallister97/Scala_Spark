import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Create a Spark Session
val spark = SparkSession.builder().getOrCreate()

// Use Spark to read in the advertising csv file
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("advertising.csv")

// Print the Schema of the DataFrame
data.printSchema()

// Get the hour from the timestamp
val timedata = data.withColumn("Hour",hour(data("Timestamp")))

// Select columns to use in analysis
// Set Clicked on Ad as label since this is what we will try and predict
val logregdata = timedata.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage",$"Hour",$"Male")

// Set up feature columns for model
val assembler = new VectorAssembler().setInputCols(Array("Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage","Hour")).setOutputCol("features")

// Use randomSplit to create a train test split of 70/30
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3))

// Create a new LogisticRegression object called lr
val lr = new LogisticRegression()

// Create a new pipeline with the stages: assembler, lr
val pipeline = new Pipeline().setStages(Array(assembler, lr))

// Fit the pipeline to training set.
val model = pipeline.fit(training)

// Get Results on Test Set with transform
val results = model.transform(test)

// Convert the test results to an RDD using .as and .rdd
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

// Instantiate a new MulticlassMetrics object
val metrics = new MulticlassMetrics(predictionAndLabels)

// Print out the Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)
