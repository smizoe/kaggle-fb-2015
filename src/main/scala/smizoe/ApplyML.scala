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
import org.apache.spark.mllib.classification.{SVMModel,SVMWithSGD}


object RFCV extends Util with S3Conf with RFConf with CVBase{
  def main(args: Array[String]): Unit={
    val conf = new SparkConf().setAppName("RandomForest-CV")
    implicit val sc = new SparkContext(conf)
    implicit val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    // possibly we can join empty rdds
    val bidsDf                  = createTable(sc.parallelize(Seq[String]()), bidsSchema)
    val biddersDf               = createTable(sc.parallelize(Seq[String]()), biddersSchema)
    val trainingDf              = innerJoinOnBidderId(biddersDf, bidsDf)
    val categoricalFeaturesInfo = makeCategoricalFeaturesInfo(trainingDf, "outcome", unnecessaryFeatureNames, allMappings, Set("device", "url"))
    val modelMaker: RDD[LabeledPoint] => RandomForestModel = RandomForest.trainClassifier(_, 2, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    runCV(1, modelMaker, false)
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
object SVMCV extends Util with S3Conf with SVMConf with CVBase {
  def main(args: Array[String]): Unit= {
    val conf = new SparkConf().setAppName("SVM-CV")
    implicit val sc = new SparkContext(conf)
    implicit val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val modelMaker: RDD[LabeledPoint] => SVMModel = SVMWithSGD.train(_, numIterations)
    runCV(1, modelMaker)
  }
}

object SVMPredict extends Util with S3Conf with SVMConf {
  def main(args: Array[String]): Unit={
    val conf = new SparkConf().setAppName("SVM-Predict")
    implicit val sc = new SparkContext(conf)
    implicit val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val bidsDf       = createTable(fromS3File(bucket, dataDir + "bids.csv.gz"), bidsSchema)
    val biddersDf    = createTable(fromS3File(bucket, dataDir + s"submission/bidders_submission.csv"), submissionSchema)
    val submissionDf = innerJoinOnBidderId(biddersDf, bidsDf)
    val vectors      = convertToVectors(submissionDf, allMappings, unnecessaryFeatureNames.toSet + "outcome")
    val model        = SVMModel.load(sc, bucket+modelDir)

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

object SVMTrain extends Util with S3Conf with SVMConf {
  def main(args: Array[String]): Unit={
    val conf = new SparkConf().setAppName("SVM-Train")
    implicit val sc = new SparkContext(conf)
    implicit val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val bidsDf                  = createTable(fromS3File(bucket, dataDir + "bids.csv.gz"), bidsSchema)
    val biddersDf               = createTable(fromS3File(bucket, dataDir + s"submission/bidders_submission_train.csv"), biddersSchema)
    val trainingDf              = innerJoinOnBidderId(biddersDf, bidsDf)
    val labeledPoints           = convertToLabeledPoints(trainingDf, "outcome", allMappings, unnecessaryFeatureNames.toSet)
    val model                   = SVMWithSGD.train(labeledPoints, numIterations)
    model.save(sc, bucket + modelDir)
    println("training succeeded")
  }
}
