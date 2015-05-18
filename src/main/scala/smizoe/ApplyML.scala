package smizoe
import scala.io.Source
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import java.net.URL

trait ContextPack {
  val sc: SparkContext
  val sqlContext: SQLContext
}
trait Util{ this: ContextPack =>
  import sqlContext.implicits._

  val dataDir = "s3://mizoe-kaggle-data/data/"

  private val mappingDir    = "s3://mizoe-kaggle-data/mappings/"
  lazy val countryMap     = getFactorMap(mappingDir + "countries.map")
  lazy val deviceMap      = getFactorMap(mappingDir + "devices.map")
  lazy val merchandiseMap = getFactorMap(mappingDir + "merchandises.map")
  lazy val urlMap         = getFactorMap(mappingDir + "urls.map")
  lazy val allMappings    = Map(
    "country" -> countryMap,
    "device" -> deviceMap,
    "merchandise" -> merchandiseMap,
    "url" -> urlMap
    )

  private val schemaDir     = "s3://mizoe-kaggle-data/schemas/"
  lazy val biddersSchema = getSchema(schemaDir + "bidders.schema")
  lazy val bidsSchema    = getSchema(schemaDir + "bids.schema")
  lazy val submissionSchema = getSchema(schemaDir + "submission.schema")

  val unnecessaryFeatureNames = Array("bidder_id", "payment_acount", "address", "bid_id", "auction", "ip")

  def createTable(csv: RDD[String], schema: StructType) = {
    val rowRDD = csv.map(_.split(",")).map(p => Row.fromSeq(p))
    sqlContext.createDataFrame(rowRDD, schema)
  }
  def convertToLabeledPoints(df: DataFrame, outcomeName: String, factorConverter: Map[String, Map[String, Double]], factorsToRemove: Set[String]): RDD[LabeledPoint] = {
    val names = df.columns
    val namesWithoutOutcome = names.filterNot( name => name == outcomeName)
    val outcomeIndx = names.indexOf(outcomeName)
    val dfWithoutOutcome = if (namesWithoutOutcome.length > 1) {
      df.select(namesWithoutOutcome(0), namesWithoutOutcome.tail: _*)
    } else {
      df.select(namesWithoutOutcome(0))
    }
    val features = convertToVectors(dfWithoutOutcome, factorConverter, factorsToRemove)
    df.rdd.zip(features).map{ case(row, vector) =>
      val outcome = row.getDouble(outcomeIndx)
      LabeledPoint(outcome, vector)
    }
  }

  def convertToVectors(df: DataFrame, factorConverter: Map[String, Map[String, Double]], factorsToRemove: Set[String]): RDD[Vector] = {
    val names = df.columns
    df.map{ r =>
      val featuresWithHoles: Array[Option[Double]] = {for( indx <- Range(0, names.length)) yield {
        val name = names(indx)
        if(factorsToRemove.contains(name))
          None
        else{
          if(factorConverter.contains(name))
            factorConverter(name).get(r.getString(indx))
          else
            Some(r.getDouble(indx))
        }
      }}.toArray
      val features = featuresWithHoles.filterNot(_.isEmpty).map(_.get)
      Vectors.dense(features)
    }
  }

  def getFactorMap(path: String): Map[String, Double] = {
    val rdd = sc.textFile(path)
    tsv2Map(rdd)
  }

  def getFactorMap(path: URL): Map[String, Double] = {
    val source = Source.fromURL(path)
    tsv2Map(source)
  }

  def getSchema(path: String): StructType = {
    val rdd = sc.textFile(path)
    parseSchema(rdd)
  }

  def getSchema(path: URL): StructType = {
    val source = Source.fromURL(path)
    parseSchema(source)
  }

  def tsv2Map(text:RDD[String]):Map[String, Double] = {
    text.map(line => {
      val Array(key, value) = line.split("\t")
      (key, java.lang.Double.parseDouble(value).doubleValue)
    }).collect().toMap
  }

  def tsv2Map(text: Source):Map[String, Double] = {
    text.getLines.map(line => {
      val Array(key, value) = line.split("\t")
      (key, java.lang.Double.parseDouble(value).doubleValue)
    }).toMap
  }


  def parseSchema(text: RDD[String]): StructType = {
    StructType {
      text.map(schemaLineProcess(_)).collect()
    }
  }
  def parseSchema(text: Source): StructType = {
    StructType {
      text.getLines.map(schemaLineProcess(_)).toArray
    }
  }

  def schemaLineProcess(line: String): StructField = {
    val Array(name, coltype) = line.split("\\s+")
    val dataType: DataType = coltype.toLowerCase match {
      case "double"  => DoubleType
      case "string"  => StringType
      case "decimal" => DecimalType(None)
      case "int"     => IntegerType
      case _         => throw new java.lang.RuntimeException("Unknown type is passed as a column type in parseSchema")
    }
    StructField(name, dataType)
  }

  def makeCategoricalFeaturesInfo(df: DataFrame, outcomeName: String, factorsToRemove: Array[String], allMaps: Map[String, Map[String, Double]]): Map[Int, Int] ={
    val names = df.columns
    val remainingFeatures = names.filterNot(name => name == outcomeName || factorsToRemove.contains(name))
    val factorNames = allMaps.keys
    factorNames.map { name: String =>
      (remainingFeatures.indexOf(name), allMaps(name).size)
    }.toMap
  }

  def innerJoinOnBidderId(biddersDf: DataFrame, bidsDf: DataFrame): DataFrame = {
    biddersDf.join(bidsDf, biddersDf("bidder_id") === bidsDf("bidder_id"), "inner")
  }


  def getAreaUnderROC(probAndOutcome: Seq[(Double, Double)]): Double = {
    val sorted    = probAndOutcome.sortBy( {case (prob: Double, outcome: Double) => prob})
    val numHumans = probAndOutcome.count( {case (prob: Double, outcome: Double) => outcome == 0.0})
    val numBots   = probAndOutcome.length - numHumans
    val xStep     = 1.0 / numHumans
    val yStep     = 1.0 / numBots

    val (xTotal, yTotal, areaUnderCurve) = sorted.foldRight((0.0, 0.0, 0.0)) { (that, acc) =>
      val (prob, outcome) = that
      val (xSubtot, ySubtot, areaSubtot) = acc
      if(outcome == 1.0)
        (xSubtot, ySubtot + yStep, areaSubtot)
      else
        (xSubtot + xStep, ySubtot, areaSubtot + xStep * ySubtot)
    }
    areaUnderCurve
  }
}

trait MLConf {
  val modelPath: String
  val cvResultDir: String
  val predictionResultDir: String
}

trait RFConf extends MLConf{
  val modelPath           = "s3://mizoe-kaggle-data/model/RF"
  val cvResultDir         = "s3://mizoe-kaggle-data/cv-result/RF"
  val predictionResultDir = "s3://mizoe-kaggle-data/submission-result/RF"
  val numTrees: Int=500
  val featureSubsetStrategy: String = "auto"
  val impurity:String = "gini"
  val maxDepth: Int=4
  val maxBins: Int = 100
}

object CVRF extends ContextPack with Util with RFConf{
  val conf = new SparkConf().setAppName("RandomForest CV")
  val sc = new SparkContext(conf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  def oneRun(trainingDf: DataFrame, validationDf: DataFrame): RDD[(Int, String, Double, Double)]= {
    val categoricalFeaturesInfo = makeCategoricalFeaturesInfo(trainingDf, "outcome", unnecessaryFeatureNames, allMappings)
    val labeledPoints = convertToLabeledPoints(trainingDf, "outcome", allMappings, unnecessaryFeatureNames.toSet)
    val model = RandomForest.trainClassifier(labeledPoints, 2, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    val validationData = convertToVectors(validationDf, allMappings, unnecessaryFeatureNames.toSet + "outcome")
    val prediction     = model.predict(validationData)
    validationDf.select("bid_id", "bidder_id", "outcome").rdd.zip(prediction).map{ case (Row(bid_id: Int, bidder_id: String, outcome: Double), pred: Double) =>
      (bid_id, bidder_id, outcome, pred)
    }
  }
  def main(args: Array[String]): Unit={
    val numValidationPair = 10

    val bidsDf = createTable(sc.textFile(dataDir + "bids.csv.gz"), bidsSchema)
    val predictions = for(indx <- Range(1, numValidationPair + 1)) yield {
      val biddersDf    = createTable(sc.textFile(dataDir + s"exploration/bidders_train_${indx}.csv"), biddersSchema)
      val validationDf = createTable(sc.textFile(dataDir + s"exploration/bidders_validation_${indx}.csv"), biddersSchema)
      val joinedTraining   = innerJoinOnBidderId(biddersDf, bidsDf)
      joinedTraining.persist()
      val joinedValidation = innerJoinOnBidderId(validationDf, bidsDf)
      val result = oneRun(joinedTraining, joinedValidation)
      joinedTraining.unpersist()
      result.map{tuple => tuple.productIterator.toArray.mkString(",")}.saveAsTextFile(cvResultDir + s"result_${indx}.csv")
      result
    }
    val areasUnderRoc = predictions.map{ result =>
      val grouped = result.groupBy { case (bid_id, bidder_id, outcome, pred) => bidder_id}
      val probAndOutcome= grouped.map { case (bidder_id, iterator) =>
        val ary      = iterator.toArray
        val totalObs = ary.length
        val numOne   = ary.count{ case (bid_id, bidder_id, outcome, pred) => pred == 1.0}
        val prob     = numOne.toDouble /totalObs
        // prob. and outcome
        (prob, ary(0)._3)
      }.collect()
      getAreaUnderROC(probAndOutcome)
    }
    val avg = areasUnderRoc.reduce( _ + _ ) / numValidationPair
    val sd  = ((areasUnderRoc.map( math.pow( _ ,  2) )).reduce(_ + _) - numValidationPair * math.pow(avg, 2)) / (numValidationPair - 1)
    println(s"Area under ROC curve of ${numValidationPair} CV sets => avg.: ${avg}, sd: ${sd}")
  }
}

object PredictRF extends ContextPack with Util with RFConf{
  val conf = new SparkConf().setAppName("RandomForest Prediction")
  val sc   = new SparkContext(conf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  def main(args: Array[String]): Unit={
    val bidsDf       = createTable(sc.textFile(dataDir + "bids.csv.gz"), bidsSchema)
    val biddersDf    = createTable(sc.textFile(dataDir + s"submission/bidders_submission.csv"), submissionSchema)
    val submissionDf = innerJoinOnBidderId(biddersDf, bidsDf)
    val vectors      = convertToVectors(submissionDf, allMappings, unnecessaryFeatureNames.toSet + "outcome")
    vectors.persist()
    val model        = RandomForestModel.load(sc, modelPath)

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
    }.map{ tuple => tuple.productIterator.toArray.mkString(",") }.saveAsTextFile(predictionResultDir)
  }
}

object TrainRF extends ContextPack with Util with RFConf{
  val conf = new SparkConf().setAppName("RandomForest Training")
  val sc = new SparkContext(conf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  def main(args: Array[String]): Unit={
    val bidsDf                  = createTable(sc.textFile(dataDir + "bids.csv.gz"), bidsSchema)
    val biddersDf               = createTable(sc.textFile(dataDir + s"submission/bidders_submission_train.csv"), biddersSchema)
    val trainingDf              = innerJoinOnBidderId(biddersDf, bidsDf)
    trainingDf.persist
    val categoricalFeaturesInfo = makeCategoricalFeaturesInfo(trainingDf, "outcome", unnecessaryFeatureNames, allMappings)
    val labeledPoints           = convertToLabeledPoints(trainingDf, "outcome", allMappings, unnecessaryFeatureNames.toSet)
    val model                   = RandomForest.trainClassifier(labeledPoints, 2, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    model.save(sc, modelPath)
    println("training succeeded")
  }
}
