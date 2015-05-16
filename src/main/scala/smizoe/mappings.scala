package smizoe
import scala.io.source
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils

object Util{
  import sqlContext.implicits._
  val sc: SparkContext
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  private val mappingDir    = "s3://mizoe-kaggle-data/mappings/"
  val countryMap    = sc.broadcast(getFactorMap(mappingDir + "countries.map"))
  val deviceMap     = sc.broadcast(getFactorMap(mappingDir + "devices.map"))
  val merchandiseMap = sc.broadcast(getFactorMap(mappingDir + "merchandises.map"))
  val urlMap        = sc.broadcast(getFactorMap(mappingDir + "urls.map"))
  val allMappings   = Map(
    "country" -> countryMap,
    "device" -> deviceMap,
    "merchandise" -> merchandiseMap,
    "url" -> urlMap
    )

  private val schemaDir     = "s3://mizoe-kaggle-data/schemas/"
  val biddersSchema = getSchema(schemaDir + "bidders.schema")
  val bidsSchema    = getSchema(schemaDir + "bids.schema")
  val submissionSchema = getSchema(schemaDir + "submission.schema")

  def createTable(csv: RDD[String], schema: StructType) = {
    val rowRDD = csv.map(_.split(",")).map(p => Row.fromSeq(p))
    sqlContext.createDataFrame(rowRDD, schema)
  }
  def convertToLabeledPoint(df: DataFrame, outcomeName: String, factorConverter: Map[String, Map[String, Double]], factorsToRemove: Set[String]): RDD[LabeledPoint] = {
    val names = df.columns
    val outcomeIndx = names.indexOf(outcomeName)
    df.map{ r =>
      val featuresWithHoles: Array[Option[Double]] = for( indx <- Range(0, names.length)) yield {
        val name = names(indx)
        if(outcomeIndx == indx || factorsToRemove.contains(name))
          None
        else{
          if(factorConverter.contains(name))
            factorConverter(name).get(r.getString(indx))
          else
            Some(r.getDouble(indx))
        }
      }.toArray
      val features = featuresWithHoles.filterNot(_.isEmpty).map(_.get)
      LabeledPoint(df.getDouble(outcomeIndx), Vectors.dense(features))
    }
  }

  def convertToVectors

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
      text.getLines.map(schemaLineProcess(_))
    }
  }

  def schemaLineProcess(line: String): DataType = {
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
}

object CV {
  import Util
  def main(args: Array[String]): Unit={
    val
  }
}

object Predict {
  def main(args: Array[String]): Unit={
  }
}

object Train{
  def main(args: Array[String]): Unit={
  }
}
