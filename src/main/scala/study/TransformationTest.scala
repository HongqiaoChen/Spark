package study


import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.Test




class TransformationTest {
  val conf = new SparkConf().setMaster("local[2]").setAppName("transformation_op")
  val sc = new SparkContext(conf)

  @Test
  def mapPartitionsTest(): Unit={
    sc.parallelize(Seq(1,2,3,4,5,6),2)
      .mapPartitions(iter=>{
        iter.map(each => each*10)
      })
      .collect()
      .foreach(item => println(item))
  }


  @Test
  def mapPartitionsWithIndex: Unit ={
    sc.parallelize(Seq(1,2,3,4,5,6),2)
      .mapPartitionsWithIndex((index,iter)=>{
        println("index:"+index)
        iter.map(item => item * 10)
      })
      .collect()
      .foreach(item=>println(item))
  }

  @Test
  def filterTest: Unit ={
    sc.parallelize(Seq(1,2,3,4,5,6,7,8,9))
      .filter(item => item % 2==0)
      .collect()
      .foreach(item => println(item))
  }

  @Test
  def sampleTest: Unit ={
    sc.parallelize(Seq(1,2,3,4,5,6,7,8,9,10))
      .sample(false,0.6)   //false 表示不放回的采样 fraction表示抽样的比例
      .collect()
      .foreach(item => println(item))
  }

  @Test
  def mapValuesTest: Unit ={
    sc.parallelize(Seq(("a",1),("b",2),("c",3)))
      .mapValues( item => item *10 )
      .collect()
      .foreach(item => println(item))
  }


  @Test
  def mapTest: Unit ={
    sc.parallelize(Seq(("a",1),("b",1),("c",1)))
      .map(item => (item._1,item._2*10))
      .collect()
      .foreach(item=>println(item))
  }

  @Test
  def intersectionTest: Unit ={
    val rdd1: RDD[Int] = sc.parallelize(Seq(1, 2, 3, 4, 5))
    val rdd2: RDD[Int] = sc.parallelize(Seq(2, 3, 4, 5, 6,7))
    rdd1.intersection(rdd2)
      .collect()
      .foreach(println(_))
  }


  /*并集运算
  由于不是集合，所以存在重复元素
  */
  @Test
  def unionTest: Unit ={
    val rdd1: RDD[Int] = sc.parallelize(Seq(1, 2, 3, 4, 5))
    val rdd2: RDD[Int] = sc.parallelize(Seq(2, 3, 4, 5, 6,7))
    rdd1.union(rdd2)
      .collect()
      .foreach(println(_))
  }

  @Test
  def subtractTest: Unit ={
    val rdd1: RDD[Int] = sc.parallelize(Seq(1, 2, 3, 4, 5))
    val rdd2: RDD[Int] = sc.parallelize(Seq(2, 3, 4, 5, 6,7))
    rdd1.subtract(rdd2)
      .collect()
      .foreach(println(_))
  }

  @Test
  def groupByKeyTest: Unit ={
    sc.parallelize(Seq(("a",1),("b",1),("c",1),("a",2)))
      .groupByKey()
      .collect()
      .foreach(println(_))
  }


  //combineByKetTest中的createCombiner初始函数只作用在第一条数据上用作打样
  @Test
  def combineByKetTest: Unit ={
    val rdd1: RDD[(String, Double)] = sc.parallelize(Seq(("hqc", 98.0), ("tx", 98.0), ("hqc", 99.0), ("tx", 96.0), ("hqc", 89.0)))
    rdd1.combineByKey(
      createCombiner = (curr:Double) => (curr,1),
      mergeValue = (curr:(Double,Int) ,nextValue:Double)=>(curr._1+nextValue,curr._2+1),
      mergeCombiners = (curr:(Double,Int),agg:(Double,Int)) =>(curr._1+agg._1,curr._2+agg._2)
    )
      .map(item => (item._1,item._2._1/item._2._2))
      .collect()
      .foreach(println(_))
  }


  // join 默认是按照key的，也即是按key进行联结
  @Test
  def joinTest: Unit ={
    val rdd1: RDD[(String, Int)] = sc.parallelize(Seq(("a", 1), ("a", 2), ("b", 1)))
    val rdd2: RDD[(String, Int)] = sc.parallelize(Seq(("a", 10), ("a", 20)))
    rdd1.join(rdd2)
      .collect()
      .foreach(println(_))
  }



  // 排序主要包括sortBy和sortByKey,sortBy所有情况都可以用（只需要提供排列的方式），sortByKey只有k-v数据可以用
  //
  @Test
  def sortTest: Unit ={
    val rdd1: RDD[Int] = sc.parallelize(Seq(2, 4, 1, 3, 5, 9, 3, 5))
    val rdd2: RDD[(String, Int)] = sc.parallelize(Seq(("a", 3), ("a", 2), ("b", 4)))

//    rdd1.sortBy(item =>item,ascending = false)
//      .collect()
//      .foreach(println(_))

    rdd2.sortBy(item=>item._2)
      .collect()
      .foreach(println(_))

//    rdd2.sortByKey()
//      .collect()
//      .foreach(println(_))
  }

  @Test
  def partitionTest(): Unit ={
    val rdd1: RDD[Int] = sc.parallelize(Seq(1, 2, 3, 4, 5), 2)

    // repartition 可以增加和减小partition的数量
//    println(rdd1.repartition(5).partitions.size)

    // coalese 如果不设置shuffle只能减小partition数量
//    println(rdd1.coalesce(5,true).partitions.size)
    println(rdd1.coalesce(1,true).partitions.size)
  }



}
