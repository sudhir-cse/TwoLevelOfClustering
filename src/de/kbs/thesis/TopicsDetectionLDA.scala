package de.kbs.thesis

import org.apache.spark.ml.clustering.KMeansSummary
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.clustering.LDAModel
import scala.collection.mutable.WrappedArray


object TopicsDetectionLDA {

	    val NUMBER_OF_FIRST_LEVEL_TOPICS = 5     //Number of clusters, value of k
			val FIRST_LEVEL_TOPIS_LENGTH = 3      //In words
			val FIRST_LEVEL_MAX_ITR = 50
			val FIRST_LEVEL_VOC_SIZE = 30
			val FIRST_LEVEL_MIN_DF = 0
			val FIRST_LEVEL_CLUSTER_MIN_PROBABILITY: Double = 0.1 //Documents belong to the topics that has distribution grater than 0.6

			val NUMBER_OF_SECOND_LEVEL_TOPICS = 4
			val SECOND_LEVEL_TOPICS_LENGTH = 5
			val SECOND_LEVEL_MAX_ITR = 20
			val SECOND_LEVEL_VOC_SIZE = 10
			val SECOND_LEVEL_MIN_DF = 0
			val SECOND_LEVEL_CLUSTER_MIN_PROBABILITY: Double = 0.1

			val topicsDFs: Array[Dataset[Row]] = new Array[Dataset[Row]](NUMBER_OF_FIRST_LEVEL_TOPICS) //To store set of documents that belong to each topic

			//Main function
			def main(args: Array[String]): Unit = {

		//Create spark session with the name 'Clustering in Archive Collection' that runs locally on all the core 
		val spark = SparkSession.builder()
				.master("local[*]")
				.appName("Clustering in Archeive Collection")
				.config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
				.getOrCreate()

				//RDD[Row(timeStemp, fileName, fileContent)]
				//File name is only the time stamp stamp
				val trainingData = spark.sparkContext.wholeTextFiles("data/KDDTraining")

				//Pre-process Dataset
				val preProcessedTrainingData = preProcessData(spark, trainingData)

				//Computes LDA models hierarchy
				computeLDAModelsHierarchy(spark, preProcessedTrainingData)

				//Prints topics hierarchy
				printTopicsHierarchy(spark)

	}

	//Pre-processing data: RDD[(String, String)]
	def preProcessData(spark: SparkSession, data: RDD[(String, String)]): Dataset[Row] = {

		//RDD[Row(timeStemp, fileName, fileContent)]
		//Filename has been composed of Timestamp and Filename. Separator as a "-" has been used
		//Data filterring includes: toLoweCasae, replace all the white space characters with single char, keep only alphabetic chars, keep only the words > 2.
		val tempData = data.map(kvTouple => Row(
				//DateTimeFormat.forPattern("YYYY-MM-dd").print(kvTouple._1.split(".")(0).toLong),
				kvTouple._1,
				"FileName", 
				kvTouple._2.toLowerCase().replaceAll("""\s+""", " ").replaceAll("""[^a-zA-Z\s]""", "").replaceAll("""\b\p{IsLetter}{1,2}\b""","")
				))

				//Convert training RDD to DataFrame(timestamp, fileName, fileContent)
				//Schema is encoded in String
				val schemaString = "timeStamp fileName fileContent"

				//Generate schema based on the string of schema
				val fields = schemaString.split(" ")
				.map ( fieldName => StructField(fieldName, StringType, nullable = true) )

				//Now create schema
				val schema = StructType(fields)

				//Apply schema to RDD
				val trainingDF = spark.createDataFrame(tempData, schema)
				//trainingDF.show()

				//split fileContent column into words
				val wordsData = new Tokenizer()
		.setInputCol("fileContent")
		.setOutputCol("words")
		.transform(trainingDF)

		//Remove stop words
		val stopWordsRemoved = new StopWordsRemover()
		.setInputCol("words")
		.setOutputCol("stopWordsFiltered")
		//.setStopWords(StopWordsRemover.loadDefaultStopWords("english"))
		.transform(wordsData)

		return stopWordsRemoved
	}

	//Computes LDA Models hierarchy 
	def computeLDAModelsHierarchy(spark: SparkSession, preProcessedDataset: Dataset[Row]): Unit = {

		//Term-frequencies vector
		//LDA requires only TF as input
		val tfModelF = new CountVectorizer()
		.setInputCol("stopWordsFiltered")
		.setOutputCol("featuresTFF")
		.setVocabSize(FIRST_LEVEL_VOC_SIZE)
		.setMinDF(FIRST_LEVEL_MIN_DF)
		.fit(preProcessedDataset)

		//save model
		tfModelF.write.overwrite().save("LDAModels/featuresSpace/firstLevelTF")

		val featurizedDataF  = tfModelF.transform(preProcessedDataset)

		featurizedDataF.cache()

		//Train LDA Model
		val ldaModelF = new LDA()
		.setK(NUMBER_OF_FIRST_LEVEL_TOPICS)
		.setMaxIter(FIRST_LEVEL_MAX_ITR)
		.setFeaturesCol("featuresTFF")
		.setTopicDistributionCol("topicDistributionF")
		.fit(featurizedDataF)

		//save LDA model
		ldaModelF.write.overwrite().save("LDAModels/firstLeveTopiclModel")

		//Save as temp view
		ldaModelF.transform(featurizedDataF).createOrReplaceTempView("firstLevelTopicsView")

		//compute the first level topics DataFrame
		val firstLevelVocalsF = tfModelF.vocabulary

		val describeTopiceDFF = ldaModelF.describeTopics(FIRST_LEVEL_TOPIS_LENGTH)

		//User defined function to compute topics name
		val udfIndicesToTopicsF = udf[Array[String], WrappedArray[Int]](indices => {

			indices.toArray.map {index => firstLevelVocalsF(index)}

		})

		//use user defined functions
		val topicsDFF = describeTopiceDFF.withColumn("firstLevelTopic", udfIndicesToTopicsF(describeTopiceDFF.col("termIndices")))

		//topicsDFF.show()

		//save as a table
		topicsDFF.createOrReplaceTempView("firstLevelTopicsView")

		/*--------------Work towards second level models and topics----------------*/

		//Filter out each topic and documents in it and store them in topicsDFs array
		val transDataSet = ldaModelF.transform(featurizedDataF)

		for (topicIndex <- 0 to NUMBER_OF_FIRST_LEVEL_TOPICS-1 ){

			val firstTopicsDocumentsSet = transDataSet.filter { row => row.getAs[DenseVector]("topicDistributionF").toArray(topicIndex) >= FIRST_LEVEL_CLUSTER_MIN_PROBABILITY }

			topicsDFs(topicIndex) = firstTopicsDocumentsSet

		} 

		/*---------- Compute second level topics model ----------*/
		topicsDFs.indices.foreach { i => {

			//Term-frequencies vector
			//LDA requires only TF as input
			val tfModelS = new CountVectorizer()
			.setInputCol("stopWordsFiltered")
			.setOutputCol("featuresTFS")
			.setVocabSize(SECOND_LEVEL_VOC_SIZE)
			.setMinDF(SECOND_LEVEL_MIN_DF)
			.fit(topicsDFs(i))

			//save model
			tfModelS.write.overwrite().save("LDAModels/featuresSpace/secondLevelTF"+i)

			val featurizedDataS  = tfModelS.transform(topicsDFs(i))

			featurizedDataS.cache()

			//Train LDA Model
			val ldaModelS = new LDA()
			.setK(NUMBER_OF_SECOND_LEVEL_TOPICS)
			.setMaxIter(SECOND_LEVEL_MAX_ITR)
			.setFeaturesCol("featuresTFS")
			.setTopicDistributionCol("topicDistributionS")
			.fit(featurizedDataS)

			//save LDA model
			ldaModelS.write.overwrite().save("LDAModels/secondLeveTopicsModel"+i)

			//save the transformation as a temp view
			ldaModelS.transform(featurizedDataS).createOrReplaceTempView("secondLevelTopicsView"+i)

			//compute the second level topics DataFrame
			val secondLevelVocalsS = tfModelS.vocabulary

			val describeTopiceDFS = ldaModelS.describeTopics(SECOND_LEVEL_TOPICS_LENGTH)

			//User defined function to compute topics name
			val udfIndicesToTopicsS = udf[Array[String], WrappedArray[Int]](indices => {

				indices.toArray.map {index => secondLevelVocalsS(index)}

			})

			//use user defined functions
			val topicsDFS = describeTopiceDFS.withColumn("secondLevelTopic", udfIndicesToTopicsS(describeTopiceDFS.col("termIndices")))

			//topicsDFF.show()

			//save as a table
			topicsDFF.createOrReplaceTempView("secondLevelTopicsView"+i)

		} 
		}	
	}

	//Prints topics hierarchy
	def printTopicsHierarchy(spark: SparkSession): Unit = {

		println("First level topic")
		spark.sql("SELECT * FROM firstLevelTopicsView").show()

	}
}
