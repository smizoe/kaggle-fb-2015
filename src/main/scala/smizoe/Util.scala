package smizoe
import scala.io.Source
import scala.math.BigDecimal
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.conf.Configuration
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import java.nio.file.Paths
import java.net.{URL,URI}

trait DataConf{
  val dataDir   :String
  val mappingDir:String
  val schemaDir :String
  implicit val fs: FileSystem
}

trait S3Conf extends DataConf{
  val bucket      = "s3://mizoe-kaggle-data"
  val dataDir     = "/data/"
  val mappingDir  = "/mappings/"
  val schemaDir   = "/schema/"
  val conf        = new Configuration()
  implicit val fs: FileSystem= FileSystem.get(new URI(bucket + dataDir),conf)
}


trait MLConf {
  val modelDir: String
  val cvResultDir: String
  val predictionResultDir: String
}

trait RFConf extends MLConf{ this: DataConf with Util =>
  val modelDir            = "/model/RF"
  val cvResultDir         = "/cv-result/RF"
  val predictionResultDir = "/submission-result/RF"

  lazy val biddersSchema = getSchema(strToHdfsPath(schemaDir, "bidders.schema"))
  lazy val bidsSchema    = getSchema(strToHdfsPath(schemaDir, "bids.schema"))
  lazy val submissionSchema = getSchema(strToHdfsPath(schemaDir, "submission.schema"))
  private val paths = Seq("countries.map", "devices.map", "merchandises.map", "urls.map").map(strToHdfsPath(mappingDir, _))
  private val names = Seq("country", "device", "merchandise", "url")
  lazy val allMappings   = genAllMappings(names, paths)
  val unnecessaryFeatureNames = Array("bidder_id", "bidder_id_left", "payment_account", "address", "bid_id", "auction", "ip")

  // ML Settings
  val numTrees: Int=30
  val featureSubsetStrategy: String = "auto"
  val impurity:String = "gini"
  val maxDepth: Int=4
  lazy val maxBins: Int = Seq(100, allMappings.values.map(_.size).max + 1).max
}


trait Util{
  def fromS3File(bucket: String, path: String)(implicit sc: SparkContext) : RDD[String] = {
    sc.textFile(bucket + path)
  }
  def strToHdfsPath(dir: String, file: String): Path = {
    new Path(Paths.get(dir, file).toString)
  }

  def genAllMappings(names: Seq[String], paths: Seq[Path])(implicit fs: FileSystem): Map[String, Map[String, Double]] = {
    val maps = paths.map(getFactorMap)
    names.zip(maps).toMap
  }

  def createTable(csv: RDD[String], schema: StructType)(implicit sqlContext: SQLContext): DataFrame = {
    import sqlContext.implicits._
    val names       = schema.toIterator.map{case StructField(name, _, _, _) => name}.toSeq
    val newColNames = names.map((name: String)=> "_" + name)
    val expressions = names.zip(newColNames).map{case (orig, newN) => s"${newN} AS ${orig}"}
    val toInt     = udf[Int, String]( _.toInt)
    val toDouble  = udf[Double, String]( _.toDouble)
    val toDecimal = udf[BigDecimal, String](BigDecimal(_))

    val rowRDD = csv.map(_.split(",")).map(p => Row.fromSeq(p))
    val df: DataFrame = sqlContext.createDataFrame(rowRDD, schema)
    val newDf: DataFrame = schema.toIterator.foldLeft(df){ case (accDf: DataFrame, StructField(name, dtype, _, _)) => {
      accDf.withColumn("_" + name, dtype match{
        case _: DoubleType.type  => toDouble(accDf(name))
        case _: StringType.type  => accDf(name)
        case _: DecimalType      => toDouble(accDf(name)) // since this is to be used in MLLib
        case _: IntegerType.type => toDouble(accDf(name)) // since this is to be used in MLLib
      })
    }}
    newDf.selectExpr(expressions.toArray :_*)
  }

  def convertToLabeledPoints(df: DataFrame, outcomeName: String, factorConverter: Map[String, Map[String, Double]],
                             factorsToRemove: Set[String], outcomeConverter: Map[String, Double] = Map()): RDD[LabeledPoint] = {
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
      val outcome =
        if(outcomeConverter.isEmpty)
          row.getDouble(outcomeIndx)
        else
          outcomeConverter(row.getString(outcomeIndx))
      LabeledPoint(outcome, vector)
    }
  }

  def convertToVectors(df: DataFrame, factorConverter: Map[String, Map[String, Double]], factorsToRemove: Set[String]): RDD[Vector] = {
    val names = df.columns
    df.map{ r =>
      val featuresWithHoles: Array[Option[Double]] = {for( indx <- Range(0, names.length)) yield {
        val name = names(indx)
        if(factorsToRemove.contains(name)) {
          None
        }else{
          if(factorConverter.contains(name)) {
            val converter = factorConverter(name)
            Some(converter.getOrElse(r.getString(indx), converter.size.toDouble))
          }else
            Some(r.getDouble(indx))
        }
      }}.toArray
      val features = featuresWithHoles.filterNot(_.isEmpty).map(_.get)
      Vectors.dense(features)
    }
  }

  def getFactorMap(path: String)(implicit sc: SparkContext): Map[String, Double] = {
    val rdd = sc.textFile(path)
    tsv2Map(rdd)
  }

  def getFactorMap(path: URL): Map[String, Double] = {
    val source = Source.fromURL(path)
    val result = tsv2Map(source)
    source.close()
    result
  }

  def getFactorMap(path: Path)(implicit fs: FileSystem): Map[String, Double] = {
    val source = Source.fromInputStream(fs.open(path))
    val result = tsv2Map(source)
    source.close()
    result
  }

  def getSchema(path: String)(implicit sc: SparkContext): StructType = {
    val rdd = sc.textFile(path)
    parseSchema(rdd)
  }

  def getSchema(path: Path)(implicit fs: FileSystem): StructType = {
    val source = Source.fromInputStream(fs.open(path))
    val result = parseSchema(source)
    source.close()
    result
  }

  def getSchema(path: URL): StructType = {
    val source = Source.fromURL(path)
    val result = parseSchema(source)
    source.close()
    result
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

  // needsDefaultValue contains the names of factors that have values which are 'unknown' to its Map;
  // that is, if feature 'feat1' is contained in needsDefaultValue, df("feat1") contains a value which
  // is not contained in allMaps("feat1") as its key
  def makeCategoricalFeaturesInfo(df: DataFrame, outcomeName: String, factorsToRemove: Array[String],
                                  allMaps: Map[String, Map[String, Double]], needsDefaultValue: Set[String]=Set()): Map[Int, Int] ={
    val names = df.columns
    val remainingFeatures = names.filterNot(name => name == outcomeName || factorsToRemove.contains(name))
    val factorNames = allMaps.keys
    factorNames.map { name: String =>
      (remainingFeatures.indexOf(name),
       if(needsDefaultValue.contains(name)) allMaps(name).size + 1 else  allMaps(name).size)
    }.toMap
  }

  def innerJoinOnBidderId(biddersDf: DataFrame, bidsDf: DataFrame): DataFrame = {
    val renamed = biddersDf.withColumnRenamed("bidder_id", "bidder_id_left")
    renamed.join(bidsDf, renamed("bidder_id_left") === bidsDf("bidder_id"))
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