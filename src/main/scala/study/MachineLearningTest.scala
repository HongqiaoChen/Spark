package study

//import ml.dmlc.xgboost4j.java.XGBoost
//import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LinearSVC, LogisticRegression, LogisticRegressionModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, ChiSqSelectorModel, HashingTF, IDF, IndexToString, OneHotEncoderEstimator, OneHotEncoderModel, StringIndexer, StringIndexerModel, Tokenizer, VectorAssembler, VectorIndexer, Word2Vec, Word2VecModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.junit.Test

import scala.util.Random

class MachineLearningTest {
  val conf = new SparkConf().setMaster("local[2]").setAppName("MachineLearningTest")
  val spark = new SparkSession.Builder().config(conf).getOrCreate()
  import spark.implicits._

  @Test
  def LinearRegressionTest(): Unit ={
    val file: DataFrame = spark.read.format("csv").option("sep", ",").option("header", true).load("./data/house.csv")
    val random = new Random()
    val frame1: DataFrame = file.select("square", "price")
    val dataset1: Dataset[(Double, Double, Double)] = frame1.map(row => (row.getAs[String](0).toDouble, row.getString(1).toDouble, random.nextDouble()))
    val dataset2: Dataset[Row] = dataset1.toDF("square", "price", "rand").sort("rand")
    val assembler: VectorAssembler = new VectorAssembler().setInputCols(Array("square")).setOutputCol("features")
    val frame2: DataFrame = assembler.transform(dataset2)
    val lr = new LogisticRegression().setLabelCol("price").setFeaturesCol("features").setRegParam(0.3).setElasticNetParam(0.8).setMaxIter(10)
    val Array(train,test): Array[Dataset[Row]] = frame2.randomSplit(Array(0.8, 0.2))
    val model: LogisticRegressionModel = lr.fit(train)
    val predict: DataFrame = model.transform(test)
    predict.show()
    spark.close()
  }

  // pipeline学习实例--语义分析
  @Test
  def PipelinesTest(): Unit ={
    // 训练数据
    val train_data =  spark.createDataFrame(Seq((0L, "a b c d e spark", 1.0),(1L, "b d", 0.0),(2L, "spark f g h", 1.0),(3L, "hadoop mapreduce", 0.0))).toDF("index","text","label")
    // 构建pipeline
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val hashingTf: HashingTF = new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features")
    val regression: LogisticRegression = new LogisticRegression().setMaxIter(10).setRegParam(0.01).setFeaturesCol("features").setLabelCol("label")
    val pipeline: Pipeline = new Pipeline().setStages(Array(tokenizer, hashingTf, regression))
    // 训练pipeline得到模型
    val model: PipelineModel = pipeline.fit(train_data)
    // 测试数据
    val test_data: DataFrame = spark.createDataFrame(Seq((4L, "spark i j spark"), (5L, "l m n"), (6L, "spark a"), (7L, "apache hadoop"))).toDF("index", "text")
    val result: DataFrame = model.transform(test_data)
    // 展示结果
    result.columns.foreach(println(_))
    result.select("index","text","probability","prediction").collect().foreach{
      case Row(id: Long, text: String, probability: Vector, prediction: Double) =>
        println(s"($id, $text) --> prob=$probability, prediction=$prediction")
    }
    spark.close()
  }

  // TF-IDF 洗数据
  @Test
  def TFIDFTest(): Unit ={
    val textdata: DataFrame = spark.createDataFrame(Seq((0, "I heard about Spark and I love Spark"), (0, "I wish Java could use case classes"), (1, "Logistic regression models are neat")))
      .toDF("label", "text")
    val tokenizer: Tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val hashingTF: HashingTF = new HashingTF().setNumFeatures(2000).setInputCol("words").setOutputCol("features")
    val idf: IDF = new IDF().setInputCol("features").setOutputCol("tfidf")
    val pipeline: Pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf))
    val model: PipelineModel = pipeline.fit(textdata)
    model.transform(textdata).select("text","tfidf").collect().foreach{
      case Row(text:String, tfidf:Vector) =>
        println(s"$text --> $tfidf")
    }
    spark.close()
  }

  // Word2Vec
  @Test
  def word2VecTest()={
    val textdata: DataFrame = spark.createDataFrame(Seq(("Hi I heard about Spark", 0L), ("I wish Java could use case classes", 1L), ("Logistic regression models are neat", 2L))).toDF("text", "index")

    val tokenizer: Tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

    val wordsdata: DataFrame = tokenizer.transform(textdata)

    val word2Vec: Word2Vec = new Word2Vec().setInputCol("words").setOutputCol("feature").setVectorSize(5).setStepSize(10).setMinCount(0)

    val word2VecModel: Word2VecModel = word2Vec.fit(wordsdata)

    val result: DataFrame = word2VecModel.transform(wordsdata)

    result.columns.foreach(x=>println(x))

    result.select("index","feature").collect().foreach{
      case Row(index:Long,feature:Vector) =>
        println(s"$index -> $feature")
    }
  }

  //StringIndexer 将字符串转化为数值
  @Test
  def StringIndexTest(): Unit ={
    var raw_data: DataFrame = spark.createDataFrame(Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))).toDF("index", "category")

    val indexer: StringIndexer = new StringIndexer().setInputCol("category").setOutputCol("ID")

    val stringIndexerModel: StringIndexerModel = indexer.fit(raw_data)

    val stringIndexerResult: DataFrame = stringIndexerModel.transform(raw_data)

    stringIndexerResult.show()
  }

  // OneHotEncoder将数值转化为独热向量
  @Test
  def oneHotEncoderTest(): Unit ={
    var raw_data: DataFrame = spark.createDataFrame(Seq((0L, "a"), (1L, "b"), (2L, "c"), (3L, "a"), (4L, "a"), (5L, "c"))).toDF("index", "category")

    val indexer: StringIndexer = new StringIndexer().setInputCol("category").setOutputCol("ID")

    val stringIndexerModel: StringIndexerModel = indexer.fit(raw_data)

    val stringIndexerResult: DataFrame = stringIndexerModel.transform(raw_data)

    val oneHotEncoderEstimator: OneHotEncoderEstimator = new OneHotEncoderEstimator().setInputCols(Array("ID")).setOutputCols(Array("OneHotEncoder")).setDropLast(false)

    val oneHotEncoderModel: OneHotEncoderModel = oneHotEncoderEstimator.fit(stringIndexerResult)

    val oneHotEncoderResult: DataFrame = oneHotEncoderModel.transform(stringIndexerResult)

    oneHotEncoderResult.show()

  }

  // ChiSqSelect卡方选择
  @Test
  def chiSqSelectTest(): Unit ={
    val data: DataFrame = spark.createDataFrame(Seq((1, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1), (2, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0), (3, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0))).toDF("index", "feature", "label")

    val chiSqSelector: ChiSqSelector = new ChiSqSelector().setNumTopFeatures(2).setFeaturesCol("feature").setLabelCol("label").setOutputCol("ChiSqSelectFeature")

    val chiSqSelectorModel: ChiSqSelectorModel = chiSqSelector.fit(data)

    val result: DataFrame = chiSqSelectorModel.transform(data)

    result.show()

  }

  // 分类算法

  @Test
  def ClassifierPipelineTest(): Unit ={
    val rawData: DataFrame = spark.read.format("csv").option("sep", ",").option("inferSchema", true).load("./data/iris.csv").toDF("feature1","feature2","feature3","feature4","label")

    val assembler: VectorAssembler = new VectorAssembler().setInputCols(Array("feature1","feature2","feature3","feature4")).setOutputCol("vectorFeatures")

    val vectorData: DataFrame = assembler.transform(rawData)

    val Array(trainSet,testSet) = vectorData.randomSplit(Array(0.7, 0.3),2020)

    val stringIndexerModel: StringIndexerModel = new StringIndexer().setInputCol("label").setOutputCol("index_label").fit(trainSet)


    val logisticRegression: LogisticRegression = new LogisticRegression().setFeaturesCol("vectorFeatures").setLabelCol("index_label").setPredictionCol("prediction")

    val pipeline: Pipeline = new Pipeline().setStages(Array(stringIndexerModel,logisticRegression))

//    val decisionTreeClassifier: DecisionTreeClassifier = new DecisionTreeClassifier().setFeaturesCol("vectorFeatures").setLabelCol("index_label").setPredictionCol("prediction")

//    val pipeline: Pipeline = new Pipeline().setStages(Array(stringIndexerModel,decisionTreeClassifier))

//    val randomForestClassifier: RandomForestClassifier = new RandomForestClassifier().setFeaturesCol("vectorFeatures").setLabelCol("index_label").setPredictionCol("prediction").setNumTrees(20)
//
//    val pipeline: Pipeline = new Pipeline().setStages(Array(stringIndexerModel, randomForestClassifier))

//    val linearSVC: LinearSVC = new LinearSVC().setFeaturesCol("vectorFeatures").setLabelCol("index_label").setPredictionCol("prediction")
//
//    val pipeline: Pipeline = new Pipeline().setStages(Array(stringIndexerModel, linearSVC))
    
//    val xGBoostClassifier: XGBoostClassifier = new XGBoostClassifier().setFeaturesCol("vectorFeatures").setLabelCol("index_label").setPredictionCol("prediction").setMaxDepth(5).setEta(0.05).setNumClass(3).setObjective("multi:softprob")
//
//    val pipeline: Pipeline = new Pipeline().setStages(Array(stringIndexerModel, xGBoostClassifier))

    val pipelineModel: PipelineModel = pipeline.fit(trainSet)

    val trainPredict: DataFrame = pipelineModel.transform(trainSet)

    val testPredict: DataFrame = pipelineModel.transform(testSet)

    testPredict.show()



    val train_acc: Double = new MulticlassClassificationEvaluator().setLabelCol("index_label").setPredictionCol("prediction").setMetricName("accuracy").evaluate(trainPredict )

    println(s"train_acc  -->  $train_acc ")

    val test_acc: Double = new MulticlassClassificationEvaluator().setLabelCol("index_label").setPredictionCol("prediction").setMetricName("accuracy").evaluate(testPredict)

    println(s"test_acc  -->  $test_acc ")


  }



  @Test
  def ClassifierTest(): Unit ={
    val rawData: DataFrame = spark.read.format("csv").option("sep", ",").option("inferSchema", true).load("./data/iris.csv").toDF("feature1","feature2","feature3","feature4","label")

    val assembler: VectorAssembler = new VectorAssembler().setInputCols(Array("feature1","feature2","feature3","feature4")).setOutputCol("vectorFeatures")

    val vectorData: DataFrame = assembler.transform(rawData)

    val Array(trainSet,testSet) = vectorData.randomSplit(Array(0.7, 0.3),2020)

    val stringIndexerModel: StringIndexerModel = new StringIndexer().setInputCol("label").setOutputCol("index_label").fit(trainSet)

    val trainStringIndexer: DataFrame = stringIndexerModel.transform(trainSet)

    val logisticRegression: LogisticRegression = new LogisticRegression().setFeaturesCol("vectorFeatures").setLabelCol("index_label").setPredictionCol("prediction")

    val logisticRegressionModel: LogisticRegressionModel = logisticRegression.fit(trainStringIndexer)

    val testStringIndexer: DataFrame = stringIndexerModel.transform(testSet)

    val predict: DataFrame = logisticRegressionModel.transform(testStringIndexer)

    predict.show()

  }

}
