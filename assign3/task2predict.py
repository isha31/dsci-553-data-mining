from pyspark.sql import SparkSession
import sys, time, json, math

COSINE_SIMILARITY_THRESHOLD = 0.01

def main():
    spark = SparkSession.builder.appName("assign3_task2").master("local[*]").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")
    sc.setSystemProperty('spark.driver.memory', '4g')
    sc.setSystemProperty('spark.executor.memory', '4g')
    
    args = sys.argv
    test_file, model_file, out_file = str(args[1]),str(args[2]),str(args[3])

    model_file = spark.read.json(model_file).rdd.map(lambda x: (x["Type"],x["ID"],x["Features"]))

    user_profile = model_file.filter(lambda x: x[0] == "User").map(lambda x : (x[1], x[2])).collectAsMap()
    business_profile = model_file.filter(lambda x: x[0] == "Business").map(lambda x : (x[1], x[2])).collectAsMap()

    test_rdd = spark.read.json(test_file).rdd \
            .map(lambda x: (x['user_id'], x['business_id'])) \
            .map(lambda x: (x[0],x[1],get_cosine_similarity(x[0],x[1], user_profile, business_profile))) \
            .filter(lambda x: x[2] >= COSINE_SIMILARITY_THRESHOLD) \
            .map( lambda x: "{"+f'"user_id": "{x[0]}", "business_id": "{x[1]}", "sim": {x[2]}'+"}" ) \

    with open(out_file, 'w+') as fp:
        fp.write("\n".join(test_rdd.collect()))
    
def get_cosine_similarity(user_id, business_id, user_profile, business_profile):
    if user_profile.get(user_id) is None or business_profile.get(business_id) is None:
        return 0
    u_key = set(user_profile[user_id])
    b_key = set(business_profile[business_id])
    return len(u_key.intersection(b_key)) / (math.sqrt(len(u_key)) * math.sqrt(len(b_key)))

if __name__ == "__main__":
    main()