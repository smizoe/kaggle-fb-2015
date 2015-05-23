name := "Apply-ML"

version := "1.0"

scalaVersion := "2.10.4"

organization := "smizoe"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10"  % "1.3.1" % "provided",
  "org.apache.spark" % "spark-mllib_2.10" % "1.3.1" % "provided",
  "org.apache.hadoop" % "hadoop-client"     % "2.4.0" % "provided",
  "org.scalatest"    % "scalatest_2.10"   % "2.2.4" % "test"
)
