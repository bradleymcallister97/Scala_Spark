import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

// Create a Spark Session Instance
val spark = SparkSession.builder().getOrCreate()

// Load the Wholesale Customers Data
val dataset = spark.read.format("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")

// Select columns for the training set
val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")

val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

val training_data = assembler.transform(feature_data).select("features")

// Create a Kmeans Model with K=3
val kmeans = new KMeans().setK(6)

// Fit that model to the training_data
val model = kmeans.fit(training_data)

// Evaluate clustering by computing Within Set Sum of Squared Errors.
val WSSSE = model.computeCost(training_data)
println(s"Within Set Sum of Squared Errors = $WSSSE")

// Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
