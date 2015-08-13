package smizoe
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.scalatest._
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.conf.Configuration
import scala.io.Source
import java.nio.file.{Paths,Files}
import java.net.URI
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.mllib.util.MLUtils

class UtilSpec extends FlatSpec with Matchers {

  trait TestDataConf extends DataConf {
    private val pathGetter = (path: String) => {
      getClass().getResource(path).getPath()
    }
    val dataDir = pathGetter("/data")
    val mappingDir = pathGetter("/mappings")
    val schemaDir = pathGetter("/schemas")
    val hdfsConf    = new Configuration()
    implicit val fs = FileSystem.getLocal(hdfsConf)
  }

  trait TestRFConf extends MLConf {this: DataConf with Util =>
    val tempPath            = Files.createTempDirectory("ML_test").toString
    val modelDir            = Paths.get(tempPath, "model").toString
    val cvResultDir         = Paths.get(tempPath, "cv-result").toString
    val predictionResultDir = Paths.get(tempPath, "prediction").toString

    val numTrees: Int = 10
    val featureSubsetStrategy: String = "auto"
    val impurity: String = "gini"
    val maxDepth: Int = 4
    val maxBins: Int = 100

    lazy val irisSchema     = getSchema(strToHdfsPath(schemaDir, "iris.schema"))
    lazy val outcomeMapping = getFactorMap(strToHdfsPath(mappingDir, "iris.map"))
  }

  object TestRFRun extends TestDataConf with TestRFConf with Util {
    def testRun(): Unit ={
      val conf = new SparkConf().setMaster("local[2]").
        setAppName("CountingSheep").
        set("spark.executor.memory", "1g")
      implicit val sc = new SparkContext(conf)
      implicit val sqlContext = new org.apache.spark.sql.SQLContext(sc)

      try{
        val irisDf        = createTable(sc.textFile("file://"  + dataDir + "/iris.csv"), irisSchema)
        val labeledPoints = convertToLabeledPoints(irisDf, "iris_class", Map(), Set(), false, outcomeMapping)

        val folds = MLUtils.kFold(labeledPoints, 10, 11)
        val accuracySeq = folds.map {
          case (training, validation) => {
            val model = RandomForest.trainClassifier(training, 3, Map(), numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
            val validationCount = validation.count
            val prediction = model.predict(validation.map(_.features))
            val numCorrect = validation.map(_.label).zip(prediction).map { case (actual, pred) =>
              if (actual == pred)
                1
              else
                0
            }.fold(0)(_ + _)
            numCorrect / validationCount.toDouble
          }
        }
        val avg = accuracySeq.reduce(_ + _) / 10.0
        val sd  = math.sqrt((accuracySeq.map(math.pow(_, 2)).reduce(_ + _) - 10 * math.pow(avg, 2)) / 9.0)
        println("avg. accuracy: " + avg.toString)
        println("sd of accuracy: " + sd.toString)
      } finally {
        sc.stop()
      }
    }
  }
  "Main method" should "run" in {
    TestRFRun.testRun()
  }
}