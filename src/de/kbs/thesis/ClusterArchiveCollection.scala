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
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}


object ClusterArchiveCollection {

	def main(args: Array[String]) {

		//Create spark session with the name 'Clustering in Archive Collection' that runs locally on all the core 
		val spark = SparkSession.builder()
		 	.master("local[*]")
			.appName("Clustering in Archeive Collection")
			.config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
			.getOrCreate()

		//RDD[Row(timeStemp, fileName, fileContent)]
		//Filename has been composed of Times tamp and Filename. Separator as a "-" has been used
		//Data filtering includes: toLoweCasae, replace all the white space characters with single char, keep only alphabetic chars, keep only the words > 2.
		val training = spark.sparkContext.wholeTextFiles("data")
		  .map(kvTouple => Row(kvTouple._1.split("=")(0), kvTouple._1.split("=")(1), kvTouple._2.toLowerCase().replaceAll("""\s+""", " ").replaceAll("""[^a-zA-Z\s]""", "").replaceAll("""\b\p{IsLetter}{1,2}\b""","")))

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
	
	  //This method computes topic from text documents collection
  //Input column "stopWordsFiltered", should be already pre-processed
  def computeTopic(dataset: Dataset[Row]): String = {
    
   var topic: String = ""
    
    //Term-frequencies vector
	 val tfModel = new CountVectorizer()
		.setInputCol("stopWordsFiltered")
		.setOutputCol("features-TF")
		.setVocabSize(100)
		.setMinDF(5)
		.fit(dataset)
		
	 val tf  = tfModel.transform(dataset)
	
  //TF-IDF vector
	val tfidfModel = new IDF()
	  .setInputCol("features-TF")
	  .setOutputCol("features-TFIDF")
	  .fit(tf)
  
	val vocab = tfModel.vocabulary
	val tfidfWeight  = tfidfModel.idf.toArray
	
	val vocabAndWeight = vocab.map { term => (term, tfidfWeight(vocab.indexOf(term))) }
  
	//Now sort by weight
	val sortedVocabAndWeight = vocabAndWeight.sortWith((tuple1, tuple2) => tuple1._2 > tuple2._2)
	
	sortedVocabAndWeight.foreach(println)
	 
	val impoTopics = sortedVocabAndWeight.map((tuple) => tuple._1)
	
	//argument to take (5) is the number of vocabularies terms used for topic
	impoTopics.take(5).foreach { term => topic = topic + " "+term }
	
	return topic
		
  }
  
}
//  //This method computes a topic
//  //Make sure that Data-frame has 'features' columns
//  def computeTopic(df: Dataset[Row], vocabularies: Array[String]): String = {
//    
//    var finalTopic:String = ""
//    
//    //Train a LDA model
//    //setK = 1, for this function to work properly, i.e. to return a single topic
//    val lda = new LDA().setK(2).setMaxIter(5)
//    val model = lda.fit(df)
//    
//    //Describe topics, parameter to describeTopics() means number of words the final topic will be composed of.
//    val topic = model.describeTopics(5)
//    topic.show()
//    
//    val  collectedTopic = topic.collect()
//    vocabularies.foreach { println }
//    
//    //Form final topic, topic will contain only one row as setK() has been set to 1.
//    for ((row) <- collectedTopic) { 
//      val topicNumber = row.get(0)
//      val terms:WrappedArray[Int] = row.get(1).asInstanceOf[WrappedArray[Int]]
//      for ((termIdx) <- terms) {
//        finalTopic += vocabularies(termIdx)+ " "
//      }
//    }
//    
//   // Show result
//   // val transformed = model.transform(df)
//   // transformed.show()
//    
//    println(finalTopic)
//    
//    return finalTopic
//    
//  }





















