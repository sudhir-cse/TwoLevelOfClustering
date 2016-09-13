/*
 * Important notes
 * 
 * In the method preProcessData() - sort the resulting DataFrame based on TimeStam column
 */



package de.kbs.thesis

import org.apache.spark.ml.clustering.LDA
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
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

import scala.collection.mutable.WrappedArray
import org.apache.spark.sql.Dataset

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.clustering.KMeansSummary
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.feature.CountVectorizerModel.CountVectorizerModelReader
import org.apache.spark.ml.feature.IDFModel

object ClusterArchiveCollection {
  
  val firstLevelClustersNum: Int = 5
  val firstLevelClusterMaxItr: Int = 50
  
  val secondLevelClustersNum: Int = 3
  val secondLevelClusterMaxInt: Int = 50
  
  def main(arg: Array[String]): Unit = {
     
  //Create spark session with the name 'Clustering in Archive Collection' that runs locally on all the core 
		val spark = SparkSession.builder()
		 	.master("local[*]")
			.appName("Clustering in Archeive Collection")
			.config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
			.getOrCreate()

		//RDD[Row(timeStemp, fileName, fileContent)]
		//Filename has been composed of Time stamp and Filename. Separator as a "-" has been used
		//Data filtering includes: toLoweCasae, replace all the white space characters with single char, keep only alphabetic chars, keep only the words > 2.
		val trainingData = spark.sparkContext.wholeTextFiles("data/KDDTraining")
		
		//Pre-process data
		val preProcessedTrainingData = preProcessData(spark, trainingData)
		
		//Compute K-means model Hierarchy
		computeModelHierarchy(spark, preProcessedTrainingData, firstLevelClustersNum, firstLevelClusterMaxItr, secondLevelClustersNum, secondLevelClusterMaxInt)
		
		computeTopicsHierarchy(spark)
		
//		//Test data
//		val testData = spark.sparkContext.wholeTextFiles("data/KDDTest")
//		
//		//Pre-process test data
//		val preProcessedTestData = preProcessData(spark, testData)
//		
//		//Perform topics detection on stream of test data
//		detectNewTopicsOverDataStream(spark, preProcessedTestData)
		  
   spark.stop()
    
  }
    
  //Pre-processing data: RDD[(String, String)]
  def preProcessData(spark: SparkSession, data: RDD[(String, String)]): Dataset[Row] = {
   
    //RDD[Row(timeStemp, fileName, fileContent)]
		//Filename has been composed of Timestamp and Filename. Separator as a "-" has been used
		//Data filterring includes: toLoweCasae, replace all the white space characters with single char, keep only alphabetic chars, keep only the words > 2.
		val tempData = data.map(kvTouple => Row("timeStamp", kvTouple._1, kvTouple._2.toLowerCase().replaceAll("""\s+""", " ").replaceAll("""[^a-zA-Z\s]""", "").replaceAll("""\b\p{IsLetter}{1,2}\b""","")))

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
		  .transform(wordsData)

    return stopWordsRemoved
  }
  
  //Compute feature space
  def computeFeatureSpace(preProcessedData: Dataset[Row], vocabSize: Int, minDF: Int): Dataset[Row] = {
    
    //Term-frequencies vector
	 val tfModel = new CountVectorizer()
		.setInputCol("stopWordsFiltered")
		.setOutputCol("featuresTF")
		.setVocabSize(vocabSize)
		.setMinDF(minDF)
		.fit(preProcessedData)
		
		tfModel.write.overwrite().save("models/featuresSpace/firstLevelTF")
		
		val featurizedData  = tfModel.transform(preProcessedData)
	  //featurizedData.show()
	
	  //TF-IDF vector
		val tfidfModel = new IDF()
		  .setInputCol("featuresTF")
		  .setOutputCol("featuresTFIDF")  //fist level topics features space
		  .fit(featurizedData)
		  
	 tfidfModel.write.overwrite().save("models/featuresSpace/firstLevelTFIDF")
		  
	 val tfidf = tfidfModel.transform(featurizedData)
  
	 return tfidf
	     
  }
  
  //This method computes topic from text documents collection
  //Input column "stopWordsFiltered", should be already pre-processed
  def computeTopic(dataset: Dataset[Row]): String = {
    
   var topic: String = ""
    
    //Term-frequencies vector
	 val tfModel = new CountVectorizer()
		.setInputCol("stopWordsFiltered")
		.setOutputCol("featuresTF")
		.setVocabSize(100)
		//.setMinDF(1)
		.fit(dataset)
		
	 val tf  = tfModel.transform(dataset)
	
  //TF-IDF vector
	val tfidfModel = new IDF()
	  .setInputCol("featuresTF")
	  .setOutputCol("featuresTFIDF")
	  .fit(tf)
  
	val vocab = tfModel.vocabulary
	val tfidfWeight  = tfidfModel.idf.toArray
	
	val vocabAndWeight = vocab.map { term => (term, tfidfWeight(vocab.indexOf(term))) }
  
	//now sort by weight
	val sortedVocabAndWeight = vocabAndWeight.sortWith((tuple1, tuple2) => tuple1._2 > tuple2._2)
	
	//sortedVocabAndWeight.foreach(println)
	 
	val impoTopics = sortedVocabAndWeight.map((tuple) => tuple._1)
	
	//argument to take (5) is the number of vocabularies terms used for topic
	impoTopics.take(5).foreach { term => topic = topic + " "+term }
	
	return topic
		
  }
  
  //computes firstLevel and secondLevel k-means model and store them in dir 'models'
  //Input data must have column named 'features'
  def computeModelHierarchy(spark: SparkSession, preProcessedDataset: Dataset[Row], firstLevelClustersNum: Int, firstLevelClustersMaxItr: Int, secondLevelClustersNum: Int, secondLevelClustersMaxItr: Int): Unit = {
    
    //Compute features space for first level topics
		val tfidfForFLT = computeFeatureSpace(preProcessedDataset, 100, 2)
    
		//cache data as K-means will make multiple iteration over data set
    tfidfForFLT.cache()
    
    //First level model
		val firstLevelKmeansModel = new KMeans()
		  .setK(firstLevelClustersNum)
		  .setInitMode("k-means||")
		  .setMaxIter(firstLevelClustersMaxItr)
		  .setFeaturesCol("featuresTFIDF")
		  .setPredictionCol("clusterPrediction")
		  .fit(tfidfForFLT)
		
	  //Save model on the file
		firstLevelKmeansModel.write.overwrite().save("models/firstLeveTopiclModel")
		
		//Load model from the file 
		//val kmeansModel = KMeansModel.load("models/firstLevelModel")
		
		val firstLevelKmeansModelSummary = firstLevelKmeansModel.summary
		
		//DataFrame with prediction column
		val firstLevelDF = firstLevelKmeansModelSummary.predictions
		firstLevelDF.show
		
		//save as temporary table to run SQL queries
		firstLevelDF.createOrReplaceTempView("firstLevelTable")
		
		println("All clusters are: ")
		//Iterate over all the clusters
		for (clusterIndex <- 0 until firstLevelClustersNum ){
		  
		  val clusterDF = spark.sql(s"SELECT * FROM firstLevelTable WHERE clusterPrediction = $clusterIndex")
		  
		  //Drop the following columns as they are not needed for computing model for second level topics: featuresTF, featuresTFIDF and clusterPrediction
		  val clusterDFWithRemovedFS = clusterDF.drop("featuresTF", "featuresTFIDF", "clusterPrediction")  //DataSet with removed features space
		  //clusterDFWithRemovedFS.show()
		  
		  //compute features space for second level topic
		  val tfidfForSLT = computeFeatureSpace(clusterDFWithRemovedFS, 50, 2)
		  
		  tfidfForSLT.cache()
		  
		  //Prepare model for second level topics
		  val secondLevelKmeansModel = new KMeans()
		    .setK(secondLevelClustersNum)
		    .setInitMode("k-means||")
		    .setMaxIter(secondLevelClustersMaxItr)
		    .setFeaturesCol("featuresTFIDF")
		    .setPredictionCol("clusterPrediction")
		    .fit(tfidfForSLT)
		    
	    //save model on file
		  secondLevelKmeansModel.write.overwrite().save(s"models/secondLeveTopiclModel/subModel_$clusterIndex")
		  
		  //Create a temporary table for each set of subtopics
		  secondLevelKmeansModel.transform(tfidfForSLT).createOrReplaceTempView(s"secondLevelTable_$clusterIndex")
		 
		}		
  }
  
  //Computes topics Hierarchy
  //This method assume that temp
  def computeTopicsHierarchy(spark: SparkSession): Dataset[Row] = {
    
    //Holds final topics tuple(mainTopic, subTopic)
    var topicsSet: Set[(String, String)] = Set()
    
    for (firstLevelClusterIndex <- 0 until firstLevelClustersNum){
      
      val firstLevelClusterDF = spark.sql(s"SELECT stopWordsFiltered FROM firstLevelTable WHERE clusterPrediction = $firstLevelClusterIndex")
      val mainTopic = computeTopic(firstLevelClusterDF)
      
      //Iterate over all its sub-clusters
      for(secondLevelClusterIndex <- 0 until secondLevelClustersNum){
        
        val secondLevelClusterDF = spark.sql(s"SELECT stopWordsFiltered FROM secondLevelTable_$firstLevelClusterIndex WHERE clusterPrediction = $secondLevelClusterIndex")
        val subTopic = computeTopic(secondLevelClusterDF)
        
        topicsSet = topicsSet+((mainTopic, subTopic))
      } 
    }
    
    println("List of First level topics are :-")
    topicsSet.foreach(println)
     
    val topicsDF = spark.createDataFrame(topicsSet.toSeq).toDF("MainTopic", "SubTopic")
    val sortedTopicsDF = topicsDF.sort("MainTopic")
    
    //save this DF as a temp table
    sortedTopicsDF.createOrReplaceTempView("topicsTable")
    
    println("Topic DataFrame: ")
    sortedTopicsDF.show(false)
   
    return topicsDF
  }
  
  //This method detects new emerging topics over stream of text data
  def detectNewTopicsOverDataStream(spark: SparkSession, preProcessedTestData: Dataset[Row]): Unit = {
    
    var tfModel = CountVectorizerModel.load("models/featuresSpace/firstLevelTF")
    var tfidfModel = IDFModel.load("models/featuresSpace/firstLevelTFIDF")
    
    //preProcessedTestData("stoprot", "")
    
//    val tfDF = tfModel.transform(preProcessedTestData)
//    val tfidf = tfidfModel.transform(tfDF)
    
    //Note**** : sort preprocessedTestData by TimeStamp column
    preProcessedTestData.foreach { row => {
      
     //val temp = row.
      
      //val rowDF = spark.createDataFrame(row.toSeq).toDF()
      
    } }
    
   
    
    
    
    
  }
  
}

/*
 * Important references:-
 * 
 * 	Tables:
 * 
 * 		Main   -   firstLevelTable
 * 
 * 		Second  -  secondLevelTable_$clusterIndex
 * 
 * 	Models:
 * 
 * 		First Level -  models/firstLeveTopiclModel
 * 
 * 		Second Level  -   models/secondLeveTopiclModel/subModel_$clusterIndex
 * 
 * 	Topics:
 * 
 *		topicsTable
 */



//+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+-----------------+
//|            fileName|         fileContent|               words|   stopWordsFiltered|          featuresTF|       featuresTFIDF|clusterPrediction|
//+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+-----------------+
//|file:/D:/Master-T...|acywrjmquteesujjs...|[acywrjmquteesujj...|[acywrjmquteesujj...|(100,[0,2,38],[36...|(100,[0,2,38],[0....|                0|
//|file:/D:/Master-T...|pruning and summa...|[pruning, and, su...|[pruning, summari...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                1|
//|file:/D:/Master-T...|mining optimized ...|[mining, optimize...|[mining, optimize...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                0|
//|file:/D:/Master-T...|mining the most i...|[mining, the, mos...|[mining, interest...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                1|
//|file:/D:/Master-T...|metacost  general...|[metacost, , gene...|[metacost, , gene...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                0|
//|file:/D:/Master-T...|fast and effectiv...|[fast, and, effec...|[fast, effective,...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                3|
//|file:/D:/Master-T...|extending naive b...|[extending, naive...|[extending, naive...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                0|
//|file:/D:/Master-T...| classificationba...|[, classification...|[, classification...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                0|
//|file:/D:/Master-T...|estimating campai...|[estimating, camp...|[estimating, camp...|(100,[0,1,2,3,5,6...|(100,[0,1,2,3,5,6...|                0|
//|file:/D:/Master-T...|generalized addit...|[generalized, add...|[generalized, add...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                0|
//|file:/D:/Master-T...|horting hatches  ...|[horting, hatches...|[horting, hatches...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                0|
//|file:/D:/Master-T...|discovering rollu...|[discovering, rol...|[discovering, rol...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                0|
//|file:/D:/Master-T...|compressed data c...|[compressed, data...|[compressed, data...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                3|
//|file:/D:/Master-T...|efficient progres...|[efficient, progr...|[efficient, progr...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                0|
//|file:/D:/Master-T...|densitybased inde...|[densitybased, in...|[densitybased, in...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                3|
//|file:/D:/Master-T...|using  knowledge ...|[using, , knowled...|[using, , knowled...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                4|
//|file:/D:/Master-T...|using association...|[using, associati...|[using, associati...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                4|
//|file:/D:/Master-T...| statistical theo...|[, statistical, t...|[, statistical, t...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                1|
//|file:/D:/Master-T...| study  support v...|[, study, , suppo...|[, study, , suppo...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                0|
//|file:/D:/Master-T...|accelerating exac...|[accelerating, ex...|[accelerating, ex...|(100,[0,1,2,3,4,5...|(100,[0,1,2,3,4,5...|                0|
//+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+-----------------+




