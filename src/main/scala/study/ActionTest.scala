package study

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.Test

class ActionTest {
  val conf = new SparkConf().setMaster("local[2]").setAppName("ActionTest")
  val sc = new SparkContext(conf)

  @Test
  def reduceTest(): Unit ={
    val rdd1: RDD[(String, Int)] = sc.parallelize(Seq(("1", 10), ("1", 20), ("2", 20)))
    val tuple: (String, Int) = rdd1.reduce((curr, agg) => ("total", agg._2 + curr._2))
    println(tuple)
  }


  // 不同于scala的foreach，但用法类似。
  @Test
  def foreachTest(): Unit ={
    sc.parallelize(Seq(1,2,3)).foreach(println(_))
  }


  //count的返回结果是一个值，用来统计全部结果
  //countByKey的返回结果是一个Map类型的容器，用来统计不同key出现的总次数
  //countByValue的返回结果是一个Map类型的容器，用来统计不同（k，v）对出现的次数
  @Test
  def countTest(): Unit ={
    val rdd1: RDD[(String, Int)] = sc.parallelize(Seq(("a", 1), ("b", 2), ("a", 3), ("c", 2)))
    println(rdd1.count())
    println(rdd1.countByKey())
    println(rdd1.countByValue())
  }

//  @Test
//  def take

}
