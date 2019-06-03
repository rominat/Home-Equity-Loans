package com.homeequityloan.radom.forest.model

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics, RegressionMetrics}
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.functions._

/**
  * Created by Romina Benitez on 05/2019
  */

object RandomForestHomeLoan {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().config("spark.master", "local").getOrCreate()

    val hmeq = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .csv("src/main/resources/hmeq.csv")


    hmeq.show()

    val df = hmeq.select(hmeq("BAD").as("label"),
      col("LOAN").as("loan"),
      col("MORTDUE").as("mortgage_due"),
      col("VALUE").as("value"),
      col("REASON").as("reason"),
      col("JOB").as("job"),
      col("YOJ").as("years_on_job"),
      col("DEROG").as("derog"),
      col("CLAGE").as("c_age"),
      col("NINQ").as("ninq"),
      col("CLNO").as("cln"),
      col("DEBTINC").as("dti"))

    df.show()
    df.printSchema()

    //Counts NaNs
    def countNaNs (c: Column): Column= {
      val check = (isnan(c))
      sum(check.cast("integer"))
    }

    //Checks for NaNs in df
    df.select(df.columns map (c => countNaNs(col(c)).alias(c)): _*).show


    // string indexing
    val indexer1 = new StringIndexer().
      setInputCol("reason").
      setOutputCol("reasonIndex").
      setHandleInvalid("keep")
    val indexed1 = indexer1.fit(df).transform(df)


    val indexer2 = new StringIndexer().
      setInputCol("job").
      setOutputCol("jobIndex").
      setHandleInvalid("keep")
    val indexed2 = indexer2.fit(indexed1).transform(indexed1)

    // one hot encodingEstimator
    val encoder = new OneHotEncoderEstimator().
      setInputCols(Array("reasonIndex", "jobIndex")).
      setOutputCols(Array("reasonVec", "jobVec"))
    val encodedDF = encoder.fit(indexed2).transform(indexed2)

    encodedDF.show(20)

    //check for nulls
    def count_not_null(c: Column, nanAsNull: Boolean = false) = {
      val check = c.isNotNull and (if (nanAsNull) not(isnan(c)) else lit(true))
      sum(check.cast("integer"))
    }
    encodedDF.select(encodedDF.columns map (c => count_not_null(col(c)).alias(c)): _*).show


    // define medians
    val mdueMedianArray = encodedDF.stat.approxQuantile("mortgage_due", Array(0.5), 0)
    val mdueMedian = mdueMedianArray(0)

    val valueMedianArray = encodedDF.stat.approxQuantile("value", Array(0.5), 0)
    val valueMedian = valueMedianArray(0)

    val yojMedianArray = encodedDF.stat.approxQuantile("years_on_job", Array(0.5), 0)
    val yojMedian = yojMedianArray(0)

    val derogMedianArray = encodedDF.stat.approxQuantile("derog", Array(0.5), 0)
    val derogMedian = derogMedianArray(0)

    val cageMedianArray = encodedDF.stat.approxQuantile("c_age", Array(0.5), 0)
    val cageMedian = cageMedianArray(0)

    val ninqMedianArray = encodedDF.stat.approxQuantile("ninq", Array(0.5), 0)
    val ninqMedian = ninqMedianArray(0)

    val clnMedianArray = encodedDF.stat.approxQuantile("cln", Array(0.5), 0)
    val clnMedian = clnMedianArray(0)

    val dtiMedianArray = encodedDF.stat.approxQuantile("dti", Array(0.5), 0)
    val dtiMedian = dtiMedianArray(0)


    // replace nulls with medians
    val filled = encodedDF.na.fill(Map(
      "mortgage_due" -> mdueMedian,
      "value" -> valueMedian,
      "years_on_job" -> yojMedian,
      "derog" -> derogMedian,
      "c_age" -> cageMedian,
      "ninq" -> ninqMedian,
      "cln" -> clnMedian,
      "dti" -> dtiMedian))

    filled.show()

    // Set the input columns as the features we want to use
    val assembler = new VectorAssembler().setInputCols(Array("loan","mortgage_due",
      "value", "years_on_job", "derog", "c_age", "ninq", "cln","dti", "reasonVec", "jobVec"))
      .setOutputCol("features")

    // Transform the DataFrame
    val output = assembler.transform(filled).select(col("label"), col("features"))

    output.show(false)


    //Splitting data. Two arrays for training and test data
    val Array(training, test) = output.select("label","features")
      .randomSplit(Array(0.7, 0.3), seed = 3845)

    // create the model
    val rf = new RandomForestClassifier()

    val classifier = rf.setImpurity("gini")
      .setMaxDepth(3).setNumTrees(20)
      .setFeatureSubsetStrategy("auto")
      .setSeed(3845)
    val model = classifier.fit(training)

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    val predictions = model.transform(test)
    model.toDebugString

    val accuracy = evaluator.evaluate(predictions)
    println("accuracy before pipeline fitting" + accuracy)

    val rm = new RegressionMetrics(
      predictions.select("prediction", "label").rdd.map(x =>
        (x(0).asInstanceOf[Double], x(1).asInstanceOf[Integer].toDouble))
    )
    println("MSE: " + rm.meanSquaredError)
    println("MAE: " + rm.meanAbsoluteError)
    println("RMSE Squared: " + rm.rootMeanSquaredError)
    println("R Squared: " + rm.r2)
    println("Explained Variance: " + rm.explainedVariance + "\n")

    //****************

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxBins, Array(25, 31))
      .addGrid(classifier.maxDepth, Array(5, 10))
      .addGrid(classifier.numTrees, Array(20, 60))
      .addGrid(classifier.impurity, Array("entropy", "gini"))
      .build()

    val steps: Array[PipelineStage] = Array(classifier)
    val pipeline = new Pipeline().setStages(steps)

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    val pipelineFittedModel = cv.fit(training)

    val predictions2 = pipelineFittedModel.transform(test)
    val accuracy2 = evaluator.evaluate(predictions2)
    println("accuracy after pipeline fitting" + accuracy2)

    println(pipelineFittedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(0))

    pipelineFittedModel
      .bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages(0)
      .extractParamMap

    val rm2 = new RegressionMetrics(
      predictions2.select("prediction", "label").rdd.map(x =>
        (x(0).asInstanceOf[Double], x(1).asInstanceOf[Integer].toDouble)))

    println("MSE: " + rm2.meanSquaredError)
    println("MAE: " + rm2.meanAbsoluteError)
    println("RMSE Squared: " + rm2.rootMeanSquaredError)
    println("R Squared: " + rm2.r2)
    println("Explained Variance: " + rm2.explainedVariance + "\n")

    //****
    import df.sparkSession.implicits._
    val predictionAndLabels1 = predictions.
      select($"prediction",$"label").
      as[(Double, Double)].rdd

    import df.sparkSession.implicits._
    val predictionAndLabels2 = predictions2.
      select($"prediction",$"label").
      as[(Double, Double)].rdd

    // Instantiate a new metrics objects
    val bMetrics = new BinaryClassificationMetrics(predictionAndLabels1)
    val mMetrics = new MulticlassMetrics(predictionAndLabels1)
    val labels = mMetrics.labels //labels returns the sequence of labels in ascending order


    // Instantiate a new metrics objects
    val bMetrics2 = new BinaryClassificationMetrics(predictionAndLabels2)
    val mMetrics2 = new MulticlassMetrics(predictionAndLabels2)
    //val labels2 = mMetrics2.labels //labels returns the sequence of labels in ascending order

    // Print out the Confusion matrix
    println("Confusion matrix:")
    println(mMetrics.confusionMatrix)

    // AUROC
    val auROC = bMetrics.areaUnderROC
    println("Area under ROC = " + auROC)

    val roc = bMetrics.roc
    println(roc)

    // Print out the Confusion matrix
    println("Confusion matrix:")
    println(mMetrics2.confusionMatrix)

    // AUROC
    val auROC2 = bMetrics2.areaUnderROC
    println("Area under ROC = " + auROC)

    val roc2 = bMetrics2.roc
    println(roc)
  }


}
