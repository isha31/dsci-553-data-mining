from json import decoder
from pyspark.sql import SparkSession
import csv, json, sys

def main():
    spark = SparkSession.builder.appName("assign2_task2").master("local[*]").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")
    sc.setSystemProperty('spark.driver.memory', '4g')
    sc.setSystemProperty('spark.executor.memory', '4g')

    generate_csv(spark)

def generate_csv(spark):
    args = sys.argv
    business_file, review_file, outfile = str(args[1]),str(args[2]),str(args[3])

    business_rdd = spark.read.json(business_file).rdd.filter(lambda x : x["state"] == "NV").map(lambda x :[x["business_id"],x["state"]])
    reviews_rdd = spark.read.json(review_file).rdd.map(lambda x :[x["business_id"],x["user_id"]])
    combined_rdd = business_rdd.join(reviews_rdd).map(lambda x: [x[1][1],x[0]])
    print("COUNT = ",combined_rdd.count())

    with open(outfile, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["user_id", "business_id"])
        writer.writerows(combined_rdd.collect())

if __name__ == "__main__":
    main()