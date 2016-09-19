package de.kbs.thesis

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.clustering.KMeansSummary
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.IDFModel
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.joda.time.format.DateTimeFormat
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.functions._
import scala.collection.mutable.WrappedArray

object TopicsDetectionLDA {
  
  val NUMBER_OF_FIRST_LEVEL_TOPICS = 5     //Number of clusters, value of k
  val FIRST_LEVEL_TOPIS_LENGTH = 5      //In words
  val FIRST_LEVEL_MAX_ITR = 50
  val FIRST_LEVEL_VOC_SIZE = 50
  val FIRST_LEVEL_MIN_DF = 10
  
  val NUMBER_OF_SECOND_LEVEL_TOPICS = 4
  val SECOND_LEVEL_TOPICS_LENGTH = 5
  val SECOND_LEVEL_MAX_ITR = 20
  val SECOND_LEVEL_VOC_SIZE = 35
  val SECOND_LEVEL_MIN_DF = 5
  
  val FIRST_LEVEL_CLUSTER_PROBABILITY = 0.5
  val SECOND_LEVEL_CLUSTER_PROBABILITY = 0.5
  
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
		    
		//Convert training RDD to DataFrame(fileName, fileContent)
		//Schema is encoded in String
		val schemaString = "timeStamp fileName fileContent"

		//Generate schema based on the string of schema
		val fields = schemaString.split(" ")
		  .map ( fieldName => StructField(fieldName, StringType, nullable = true) )

		//Now create schema
		val schema = StructType(fields)

		//Apply schema to RDD
		val trainingDF = spark.createDataFrame(tempData, schema)
		trainingDF.show()

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
   val tfModel = new CountVectorizer()
  	.setInputCol("stopWordsFiltered")
  	.setOutputCol("featuresTF")
  	.setVocabSize(FIRST_LEVEL_VOC_SIZE)
  	.setMinDF(FIRST_LEVEL_MIN_DF)
  	.fit(preProcessedDataset)
  	
  	//save model
  	tfModel.write.overwrite().save("LDAModels/featuresSpace/firstLevelTF")
  	
  	val featurizedData  = tfModel.transform(preProcessedDataset)
  	
  	featurizedData.cache()
  	
  	//Train LDA Model
  	val ldaModel = new LDA()
     .setK(NUMBER_OF_FIRST_LEVEL_TOPICS)
     .setMaxIter(FIRST_LEVEL_MAX_ITR)
     .setFeaturesCol("featuresTF")
     .setTopicDistributionCol("topicDistribution")
     .fit(featurizedData)
  	
    //save LDA model
    ldaModel.write.overwrite().save("LDAModels/firstLeveTopiclModel")
    
    val firstLevelVocals = tfModel.vocabulary
   
    val describeTopiceDF = ldaModel.describeTopics(FIRST_LEVEL_TOPIS_LENGTH)
    
    //User defined function
    val udfIndicesToTopics = udf[Array[String], WrappedArray[Int]](indices => {
      
     val result = new Array[String](indices.toArray.length)
     var resultIndex = 0
     
     indices.toArray.foreach{ vocalIndex => {
       
       result(resultIndex) = firstLevelVocals(vocalIndex)
       resultIndex = resultIndex + 1
       
     } }
      
     result
     
    })
    
   //use of user defined function
   val topicsDF = describeTopiceDF.withColumn("firstLevelTopic", udfIndicesToTopics(describeTopiceDF.col("termIndices")))
   
   //save as a table
   topicsDF.createOrReplaceTempView("firstLevelTopicsTable")
   
   //work for second level topics
   
   
   
    
    
    
    
    
    
//    //user defined function
//    val fromIndicesToTerms = udf[Array[String], WrappedArray[Int]](termIndices => {
//      
//      val termIndicesToArray = termIndices.toArray
//      
//      val result = new Array[String](termIndicesToArray.length)
//      var resultIndex = 0
//      
//      termIndicesToArray.foreach { termIndex => {result(resultIndex) = firstLevelVocals(termIndex); resultIndex = resultIndex + 1; } }
//      
//      result
//       
//    })
//    
//    
//    val describeTopicsDF = ldaModel.describeTopics(FIRST_LEVEL_TOPIS_LENGTH)
//    
//    val firstLevelTopics = describeTopicsDF.withColumn("topic", fromIndicesToTerms(describeTopicsDF.col("termIndices")))
//    
//    println("Topics are: ")
//    firstLevelTopics.show()
//    describeTopicsDF.show(false)
//    
    //ldaModel.describeTopics(FIRST_LEVEL_TOPIS_LENGTH).show
    
//    println("Topics Matrix:  ")
//    val topicsMatrix = ldaModel.topicsMatrix
//    for(i <- 0 until topicsMatrix.numRows){
//      for(j <- 0 until topicsMatrix.numCols){
//        print(topicsMatrix.apply(i, j)+" ")
//      }
//      print("\n")
//    }
    
    
   //val temp1 = tempView.rdd.map { row => println)}
   //temp1.foreach { x => println } 
  }
  
  //Convert any to Double type
  def toDouble: (Any) => Double = { case i: Int => i case f: Float => f case d: Double => d }
}










