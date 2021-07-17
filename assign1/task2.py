from collections import defaultdict
from pyspark.sql import SparkSession
import json, sys

def spark_impl(business_file,review_file,n):
    spark = SparkSession.builder.appName("assign1_task2").master("local[*]").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")
    business_rdd = spark.read.json(business_file).rdd
    reviews_rdd = spark.read.json(review_file).rdd
    
    business_rdd = business_rdd.map(lambda x: [x["business_id"],"" if x["categories"] is None else x["categories"]])
    reviews_rdd = reviews_rdd.map(lambda x :[x["business_id"],x["stars"]])
    combined_rdd = business_rdd.join(reviews_rdd)

    combined_rdd = combined_rdd.flatMap(lambda x : (map(lambda y: (y.strip(), (x[1][1],1)),x[1][0].split(","))))
    combined_rdd = combined_rdd.reduceByKey(lambda x,y: (x[0] + y[0],x[1]+y[1])).mapValues(lambda x: x[0]/x[1]).sortBy(lambda x: (-x[1],x[0]))
    return list(map(lambda x: list((x[0],round(x[1],1))),combined_rdd.take(n)))

def no_spark_impl(business_file,review_file,n):
    business_data = [json.loads(data) for data in open(business_file, 'r')]
    review_data = [json.loads(data) for data in open(review_file, 'r')]

    combined_data = defaultdict(lambda: {})
    result_data = defaultdict(lambda: defaultdict(int))
    for i in business_data:
        combined_data[i["business_id"]]["category"] = "" if i["categories"] is None else i["categories"]
    for i in review_data:
        combined_data[i["business_id"]]["stars_sum"] = combined_data[i["business_id"]].get("stars_sum",0)
        combined_data[i["business_id"]]["count"] = combined_data[i["business_id"]].get("count",0)
        combined_data[i["business_id"]]["stars_sum"] += i["stars"] 
        combined_data[i["business_id"]]["count"] += 1
    for key,value in combined_data.items():
        if value.get('category') is not None:
            categories = value["category"].split(",")
            for cat in categories:
                cat = cat.strip()
                if value.get('stars_sum') is not None and value.get('count') is not None:
                    result_data[cat]['stars_sum'] += value['stars_sum']
                    result_data[cat]['count'] += value['count']
    return [[i[0],round(i[1],1)] for i in sorted([[k,v["stars_sum"]/v["count"]] for k,v in result_data.items()],key = lambda x: (-x[1],x[0]))][:n]

def main():
    output = {"result":[]}
    args = sys.argv
    review_file,business_file,out_file,if_spark,n = str(args[1]),str(args[2]),str(args[3]),str(args[4]),int(args[5])

    if if_spark == "spark":
        output["result"] = spark_impl(business_file,review_file,n)
    elif if_spark == "no_spark":
        output["result"] = no_spark_impl(business_file,review_file,n)
    
    with open(out_file,'w') as outfile:
        json.dump(output,outfile)

if __name__ == "__main__":
    main()