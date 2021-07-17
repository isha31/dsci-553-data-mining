from pyspark.sql import SparkSession
import sys,json

def custom_partition(key,n_partitions):
    return (sum([ord(i) for i in str(hash(key))])) % n_partitions

def get_partition_len(iter):
    yield sum(1 for _ in iter)

def main():
    args = sys.argv
    inp_file,out_file,partition_type,n_partitions,n = str(args[1]),str(args[2]),str(args[3]),int(args[4]),int(args[5])
    output = dict()

    spark = SparkSession.builder.appName("assign1_task3").master("local[*]").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")
    sc.setSystemProperty('spark.driver.memory', '4g')
    sc.setSystemProperty('spark.executor.memory', '4g')

    reviews_rdd = spark.read.json(inp_file).rdd.map(lambda x: [x["business_id"],1])
    if partition_type == 'customized':
        reviews_rdd = reviews_rdd.partitionBy(n_partitions,lambda x :custom_partition(x[0],n_partitions))

    output["n_partitions"] = reviews_rdd.getNumPartitions()
    output["n_items"] = reviews_rdd.mapPartitions(get_partition_len, True).collect()
    reviews_rdd = reviews_rdd.reduceByKey(lambda x,y : x + y).filter(lambda x : x[1] > n)

    output["result"] = list(map(lambda x: list(x),reviews_rdd.collect()))
    with open(out_file,'w') as outfile:
        json.dump(output,outfile)


if __name__ == "__main__":
    main()