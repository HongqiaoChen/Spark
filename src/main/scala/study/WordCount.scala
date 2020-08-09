package study

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


object WordCount {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[2]").setAppName("WordCount")
    val sc  = new SparkContext(conf)

    val rdd1: RDD[String] = sc.textFile("./data/word.txt")

    val rdd2: RDD[String] = rdd1.flatMap(item => item.split(" "))

    val rdd3: RDD[(String, Int)] = rdd2.map(item => (item, 1))

    val rdd4: RDD[(String, Int)] = rdd3.reduceByKey((curr, agg) => curr + agg)

    val result: Array[(String, Int)] = rdd4.collect()

    result.foreach(item => println(item))

    sc.stop()

  }
}
