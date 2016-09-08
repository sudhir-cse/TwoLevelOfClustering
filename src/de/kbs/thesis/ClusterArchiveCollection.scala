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

}

//fit a CountVectorizerModel from the carpus
//	 val countVectorizerModel = new CountVectorizer()
//		.setInputCol("stopWordsFiltered")
//		.setOutputCol("features")
//		.setVocabSize(200)
//		.setMinDF(5)
//		.fit(stopWordsRemoved)
//		
//		val featurizedData = countVectorizerModel.transform(stopWordsRemoved)
//		
//	  //featurizedData.show()
//
//    // Trains a LDA model.
//    val lda = new LDA().setK(10).setMaxIter(10)
//    val model = lda.fit(featurizedData)
//
////    val ll = model.logLikelihood(dataset)
////    val lp = model.logPerplexity(dataset)
////    println(s"The lower bound on the log likelihood of the entire corpus: $ll")
////    println(s"The upper bound bound on perplexity: $lp")
//
//    // Describe topics.
//    val topics = model.describeTopics(3)
//    println("The topics described by their top-weighted terms:")
//    topics.show(false)
//
//    // Shows the result.
//    val transformed = model.transform(featurizedData)
//    transformed.show()
//    
//    println("----------------list of topics----------------")
//    val vocab = countVectorizerModel.vocabulary
//    for ((row) <- topics) {
//        val topicNumber = row.get(0)
////        val termIndices = row.getAs[Array[Int]](1)
////        termIndices.foreach { indeces => print("TopicName: "+ vocab(indeces)+" ") }
////        print("\n")
//    val terms:WrappedArray[Int] = row.get(1).asInstanceOf[WrappedArray[Int]]
//    for ((termIdx) <- terms) {
//        print(vocab(termIdx)+ " ")
//        }
//    print("\n")
//    }






















