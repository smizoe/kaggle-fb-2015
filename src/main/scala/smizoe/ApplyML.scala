package smizoe
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.mllib.regression.LabeledPoint


object RFCV extends Util with S3Conf with RFConf with CVBase{

//  def oneRun(trainingDf: DataFrame, validationDf: DataFrame): RDD[(Double, String, Double, Double)]= {
//    val labeledPoints = convertToLabeledPoints(trainingDf, "outcome", allMappings, unnecessaryFeatureNames.toSet)
//    val augmented = labeledPoints// ++ augmentALabel(labeledPoints, 1.0, 10)
//    val model = RandomForest.trainClassifier(augmented, 2, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
//    val validationData = convertToVectors(validationDf, allMappings, unnecessaryFeatureNames.toSet + "outcome")
//    val prediction     = model.predict(validationData)
//    validationDf.select("bid_id", "bidder_id", "outcome").rdd.zip(prediction).map{ case (Row(bid_id: Double, bidder_id: String, outcome: Double), pred: Double) =>
//      (bid_id, bidder_id, outcome, pred)
//    }
//  }
  def main(args: Array[String]): Unit={
    val conf = new SparkConf().setAppName("RandomForest-CV")
    implicit val sc = new SparkContext(conf)
    implicit val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    // possibly we can join empty rdds
    val bidsDf                  = createTable(fromS3File(bucket, dataDir + "bids.csv.gz"), bidsSchema)
    val biddersDf               = createTable(fromS3File(bucket, dataDir + s"submission/bidders_submission_train.csv"), biddersSchema)
    val trainingDf              = innerJoinOnBidderId(biddersDf, bidsDf)
    val categoricalFeaturesInfo = makeCategoricalFeaturesInfo(trainingDf, "outcome", unnecessaryFeatureNames, allMappings, Set("device", "url"))
    val modelMaker: RDD[LabeledPoint] => RandomForestModel = RandomForest.trainClassifier(_, 2, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    runCV(1, modelMaker)
  //    val numValidationPair = 1
//    implicit val sqlContext = new org.apache.spark.sql.SQLContext(sc)

//    val bidsDf = createTable(fromS3File(bucket, dataDir + "bids.csv.gz"), bidsSchema)
//    val predictions = for(indx <- Range(1, numValidationPair + 1)) yield {
//      val biddersDf    = createTable(fromS3File(bucket, dataDir + s"exploration/bidders_train_${indx}.csv"), biddersSchema)
//      val validationDf = createTable(fromS3File(bucket, dataDir + s"exploration/bidders_validation_${indx}.csv"), biddersSchema)
//      val joinedTraining   = innerJoinOnBidderId(biddersDf, bidsDf)
//      val joinedValidation = innerJoinOnBidderId(validationDf, bidsDf)
//      val categoricalFeaturesInfo = makeCategoricalFeaturesInfo(joinedTraining, "outcome", unnecessaryFeatureNames, allMappings, Set("device", "url"))
//      val result = oneRun[RandomForestModel](joinedTraining, joinedValidation, modelMaker, false)
//      result.map{tuple => tuple.productIterator.toArray.mkString(",")}.saveAsTextFile(bucket + cvResultDir + s"/result_${indx}")
//      result
//    }
//    val areasUnderRoc = predictions.map{ result =>
//      val grouped = result.groupBy { case (bid_id, bidder_id, outcome, pred) => bidder_id}
//      val probAndOutcome= grouped.map { case (bidder_id, iterator) =>
//        val ary      = iterator.toArray
//        val totalObs = ary.length
//        val numOne   = ary.count{ case (bid_id, bidder_id, outcome, pred) => pred == 1.0}
//        val prob     = numOne.toDouble /totalObs
//        // prob. and outcome
//        (prob, ary(0)._3)
//      }.collect()
//      getAreaUnderROC(probAndOutcome)
//    }
//    val avg = areasUnderRoc.reduce( _ + _ ) / numValidationPair
//    val sd  = math.sqrt(((areasUnderRoc.map( math.pow( _ ,  2) )).reduce(_ + _) - numValidationPair * math.pow(avg, 2)) / (numValidationPair - 1))
//    println(s"Area under ROC curve of ${numValidationPair} CV sets => avg.: ${avg}, sd: ${sd}")
  }
}

object RFPredict extends Util with S3Conf with RFConf{
  def main(args: Array[String]): Unit={
    val conf = new SparkConf().setAppName("RandomForest-Predict")
    implicit val sc = new SparkContext(conf)
    implicit val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val bidsDf       = createTable(fromS3File(bucket, dataDir + "bids.csv.gz"), bidsSchema)
    val biddersDf    = createTable(fromS3File(bucket, dataDir + s"submission/bidders_submission.csv"), submissionSchema)
    val submissionDf = innerJoinOnBidderId(biddersDf, bidsDf)
    val vectors      = convertToVectors(submissionDf, allMappings, unnecessaryFeatureNames.toSet + "outcome")
    val model        = RandomForestModel.load(sc, bucket+modelDir)

    val predictionResult                 = model.predict(vectors)
    val rawResult: RDD[(String, Double)] = submissionDf.select("bidder_id").rdd.zip(predictionResult).map{ case (Row(bidder_id: String), result) =>
      (bidder_id, result)
    }

    (rawResult.groupBy{case (bidder_id, result) => bidder_id}).map { case (bidder_id, iterator) =>
      val ary    = iterator.toArray
      val numObs = ary.length
      val numOne = ary.count( v => v == 1.0)
      val prob   = (numOne.toDouble) / numObs
      (bidder_id, prob)
    }.map{ tuple => tuple.productIterator.toArray.mkString(",") }.saveAsTextFile(bucket + predictionResultDir)
  }
}

object RFTrain extends Util with S3Conf with RFConf {
  def main(args: Array[String]): Unit={
    val conf = new SparkConf().setAppName("RandomForest-Train")
    implicit val sc = new SparkContext(conf)
    implicit val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val bidsDf                  = createTable(fromS3File(bucket, dataDir + "bids.csv.gz"), bidsSchema)
    val biddersDf               = createTable(fromS3File(bucket, dataDir + s"submission/bidders_submission_train.csv"), biddersSchema)
    val trainingDf              = innerJoinOnBidderId(biddersDf, bidsDf)
    val categoricalFeaturesInfo = makeCategoricalFeaturesInfo(trainingDf, "outcome", unnecessaryFeatureNames, allMappings, Set("device", "url"))
    val labeledPoints           = convertToLabeledPoints(trainingDf, "outcome", allMappings, unnecessaryFeatureNames.toSet, false)
    val model                   = RandomForest.trainClassifier(labeledPoints, 2, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    model.save(sc, bucket + modelDir)
    println("training succeeded")
  }
}
