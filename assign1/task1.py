from pyspark.sql import SparkSession
import datetime, re, sys, json

def main():
    spark = SparkSession.builder.appName("assign1_task1").master("local[*]").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")
    sc.setSystemProperty('spark.driver.memory', '4g')
    sc.setSystemProperty('spark.executor.memory', '4g')

    args = sys.argv
    inp_file,out_file,stopwords,y,m,n = str(args[1]),str(args[2]),str(args[3]),int(args[4]),int(args[5]),int(args[6])
    reviews_rdd = spark.read.json(inp_file).rdd
    stopwords = set(sc.textFile(stopwords).collect())
    stopwords.add("")
    output = dict()

    # Q1: 
    pairs = reviews_rdd.map(lambda x: (x["review_id"],1))
    output["A"] = pairs.count()

    # Q2:
    yearly_review = reviews_rdd.filter(lambda x : datetime.datetime.strptime(x["date"], '%Y-%m-%d %H:%M:%S').year == y)
    output["B"] = yearly_review.count()

    # Q3:
    pairs = reviews_rdd.map(lambda x: (x["user_id"],1)).reduceByKey(lambda x,y: x + y)
    output["C"] = pairs.count()

    # Q4:
    pairs = reviews_rdd.map(lambda x: (x["user_id"],1)).reduceByKey(lambda x,y: x + y).sortBy(lambda x: (-x[1],x[0]))
    output["D"] = list(map(lambda x: list(x),pairs.take(m)))

    # Q5:
    punctuations = r'[\(\[,\].!?:;\)]'
    filtered_rdd = reviews_rdd.flatMap(lambda x:re.sub(punctuations, " ",x["text"].lower()).split()).filter(lambda x : x not in stopwords)
    filtered_rdd = filtered_rdd.map(lambda x : (x,1)).reduceByKey(lambda x,y: x + y).sortBy(lambda x: x[1],ascending=False)
    output["E"] = list(map(lambda x: x[0],filtered_rdd.take(n)))

    with open(out_file,'w') as outfile:
        json.dump(output,outfile)

if __name__ == "__main__":
    main()