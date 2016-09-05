package de.kbs.thesis

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType


object ClusterArchiveCollection {

	def main(args: Array[String]) {

		//Create spark session with the name 'Clustering in Archive Collection' that runs locally on all the core 
		val spark = SparkSession.builder()
		 	.master("local[*]")
			.appName("Clustering in Archeive Collection")
			.config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
			.getOrCreate()

		//RDD[Row(timeStemp, fileName, fileContent)]
		//Filename has been composed of Timestamp and Filename. Separator as a "-" has been used
		val training = spark.sparkContext.wholeTextFiles("data")
		  .map(kvTouple => Row(kvTouple._1.split("-")(0), kvTouple._1.split("=")(1), kvTouple._2))

		//Convert training RDD to DataFrame(timeStamp, fileName, fileContent)
		//Schema is encoded in String
		val schemaString = "timeStamp fileName fileContent"

		//Generate schema based on the string of schema
		val fields = schemaString.split(" ")
		  .map ( fieldName => StructField(fieldName, StringType, nullable = true) )

		//Now create schema
		val schema = StructType(fields)

		//Apply schema to RDD
		//This is input for pipeline
		val trainingDF = spark.createDataFrame(training, schema)

		//split fileContent column into words
		val wordsData = new Tokenizer()
		  .setInputCol("fileContent")
		  .setOutputCol("words")
		  .transform(trainingDF)

		//Remove stop words
		val stopWordsRemoved = new StopWordsRemover()
		  .setInputCol("words")
		  .setOutputCol("stopWordsFiltered")
		  .transform(wordsData)

		//Compute term-frequency
		val featurizedData = new HashingTF()
		  .setInputCol("stopWordsFiltered")
		  .setOutputCol("rawFeatures")
		  .setNumFeatures(50)
		  .transform(stopWordsRemoved)

		//Inverse document frequency
		val rescaledData = new IDF()
		  .setInputCol("rawFeatures")
		  .setOutputCol("features")
		  .fit(featurizedData)
		  .transform(featurizedData)
		
//		rescaledData.printSchema()
//		rescaledData.select("fileName", "features").take(3).foreach { println }
		
		  rescaledData.cache()
		  
		//k-means
		val kmeans = new KMeans()
		  .setK(3)
		  .setInitMode("k-means||")
		  .setMaxIter(100)
		
		//Model corresponding to first level topics
		val firstLevelModel = kmeans.fit(rescaledData)
		
		val temp = firstLevelModel.transform(rescaledData)
		
		temp.printSchema()
		
		//print the cost
		print("With in set sum of squired errors:  "+firstLevelModel.computeCost(rescaledData))
		
		
		//save the model
		//firstLevelModel.write.overwrite().save("models/firstLevelModel")
		
		//rescaledData.printSchema()

		spark.stop()

	}

}
