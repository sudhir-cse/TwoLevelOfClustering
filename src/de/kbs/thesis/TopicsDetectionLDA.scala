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


object TopicsDetectionLDA {
  
  val NUMBER_OF_FIRST_LEVEL_TOPICS = 5     //Number of clusters, value of k
  val FIRST_LEVEL_TOPIS_LENGTH = 3      //In words
  val FIRST_LEVEL_MAX_ITR = 50
  val FIRST_LEVEL_VOC_SIZE = 50
  val FIRST_LEVEL_MIN_DF = 10
  val FIRST_LEVEL_CLUSTER_MIN_PROBABILITY: Double = 0.6 //Documents belong to the topics that has distribution grater than 0.6
  
  val NUMBER_OF_SECOND_LEVEL_TOPICS = 4
  val SECOND_LEVEL_TOPICS_LENGTH = 5
  val SECOND_LEVEL_MAX_ITR = 20
  val SECOND_LEVEL_VOC_SIZE = 35
  val SECOND_LEVEL_MIN_DF = 5
  val SECOND_LEVEL_CLUSTER_MIN_PROBABILITY: Double = 0.6
  
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
    
    //Perform final transformation
    ldaModel.transform(featurizedData).createOrReplaceTempView("tempTable")
    val topicDistributionDF = spark.sql("SELECT topicDistribution FROM tempTable");
   // topicDistributionDF.show(false)
    
    /*---------Filter out each topic and documents in it and store them in topicsDFs array------------*/
   val transDataSet = ldaModel.transform(featurizedData)
   
   for (topicIndex <- 0 to NUMBER_OF_FIRST_LEVEL_TOPICS-1 ){
     
     val firstTopicsDocumentsSet = transDataSet.filter { row => row.getAs[DenseVector]("topicDistribution").toArray(topicIndex) >= FIRST_LEVEL_CLUSTER_MIN_PROBABILITY }
     
     topicsDFs(topicIndex) = firstTopicsDocumentsSet
     
   } 
   
   /*---------- Compute second level topics ----------*/
  // topicsDFs.foreach { x => ??? }
    
//    val firstLevelVocals = tfModel.vocabulary
//   
//   val describeTopiceDF = ldaModel.describeTopics(FIRST_LEVEL_TOPIS_LENGTH)
//    
//    //print("Describe topics");
//    //describeTopiceDF.show();
//    
//    //User defined function
//    val udfIndicesToTopics = udf[Array[String], WrappedArray[Int]](indices => {
//      
//      indices.toArray.map {index => firstLevelVocals(index)}
//      
//    })
//     
//   //use user defined function
//   val topicsDF = describeTopiceDF.withColumn("firstLevelTopic", udfIndicesToTopics(describeTopiceDF.col("termIndices")))
//   
//   topicsDF.show()
//   
//   //save as a table
//   topicsDF.createOrReplaceTempView("firstLevelTopicsTable")
//   
//   //work for second level topics
//   
//       
////    println("Topics Matrix:  ")
////    val topicsMatrix = ldaModel.topicsMatrix
////    for(i <- 0 until topicsMatrix.numRows){
////      for(j <- 0 until topicsMatrix.numCols){
////        print(topicsMatrix.apply(i, j)+" ")
////      }
////      print("\n")
////    }
    
  }
}


//final schema : DataFrame only with file content will be enough as well