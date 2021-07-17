from pyspark.sql import SparkSession
import json, time, sys, math
from collections import OrderedDict

N = 3

def main():
    start = time.time()
    spark = SparkSession.builder.appName("assign3_task1").master("local[*]").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")
    sc.setSystemProperty('spark.driver.memory', '4g')
    sc.setSystemProperty('spark.executor.memory', '4g')

    args = sys.argv
    train_file, test_file, model_file, output_file, cf_type = str(args[1]), str(args[2]), str(args[3]), str(args[4]), str(args[5])

    input_rdd = spark.read.json(train_file).rdd.map(lambda x: (x["business_id"],x["user_id"],x["stars"]))

    users_dict = input_rdd.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
    reverse_users_dict = {v:k for k,v in users_dict.items()}

    business_dict = input_rdd.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
    reverse_business_dict = {v:k for k,v in business_dict.items()}
    ratings = None
    avg_rating = input_rdd.map(lambda x: (x[2],1)).reduce(lambda x,y: (x[0] + y[0], x[1] + y[1]))
    avg_rating = avg_rating[0] / avg_rating[1]

    if cf_type == "item_based":

        bus_sim = sc.textFile(model_file).map(lambda x: json.loads(x)) \
            .map(lambda x: (tuple(sorted((business_dict[x["b1"]],business_dict[x["b2"]]))),x["sim"])) \
            .collectAsMap()

        avg_bid_map = input_rdd.map(lambda x: (x[0],x[2])) \
            .groupByKey() \
            .mapValues(lambda x: list(x)) \
            .flatMap(lambda x: [(x[0], (i, 1)) for i in x[1]]) \
            .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
            .mapValues(lambda x: x[0]/x[1]).collectAsMap()

        test_rdd = spark.read.json(test_file).rdd.map(lambda x: (x["user_id"],x["business_id"])) \
            .filter(lambda x: x[0] in users_dict and x[1] in business_dict).map(lambda x: (users_dict[x[0]],business_dict[x[1]]))

        ub_map = input_rdd.map(lambda x: (users_dict[x[1]],(business_dict[x[0]],x[2]))) \
            .groupByKey() \
            .map(lambda x:( x[0] ,dict(x[1])))
        ratings = test_rdd.leftOuterJoin(ub_map) \
            .map(lambda x: (x[0],x[1][0], get_item_based_rating(x[0],x[1][0],x[1][1], bus_sim, reverse_business_dict, avg_bid_map,avg_rating))) \
            .map(lambda x: "{"+f'"user_id": "{reverse_users_dict[x[0]]}", "business_id": "{reverse_business_dict[x[1]]}", "stars": {x[2]}'+"}")

    elif cf_type == "user_based":

        user_sim = sc.textFile(model_file) \
            .map(lambda x: json.loads(x)) \
            .map(lambda x: (tuple(sorted((users_dict[x["u1"]],users_dict[x["u2"]]))),x["sim"])) \
            .collectAsMap()

        test_rdd = spark.read.json(test_file).rdd \
            .map(lambda x: (x["user_id"],x["business_id"])) \
            .filter(lambda x: x[0] in users_dict and x[1] in business_dict)
        bu_map = input_rdd.map(lambda x: (business_dict[x[0]],(users_dict[x[1]],x[2]))) \
            .groupByKey() \
            .mapValues(lambda x:dict(x)) \
            .collectAsMap()
        ub_map = input_rdd.map(lambda x: (users_dict[x[1]],(business_dict[x[0]],x[2]))) \
            .groupByKey() \
            .mapValues(lambda x:dict(x))
        avg_user_dict = ub_map.flatMap(lambda x: ([(x[0],(v,1)) for k,v in x[1].items()])) \
            .reduceByKey(lambda x,y:(x[0] + y[0], x[1] + y[1])) \
            .collectAsMap()
        ub_map =ub_map.collectAsMap()

        ratings = test_rdd.map(lambda x: (x[0],x[1],get_user_based_rating(users_dict[x[0]],business_dict[x[1]],bu_map,ub_map, user_sim, avg_user_dict,avg_rating,reverse_users_dict,users_dict))) \
            .map(lambda x: "{"+f'"user_id": "{x[0]}", "business_id": "{x[1]}", "stars": {x[2]}'+"}")

    with open(output_file, 'w+',encoding='utf-8') as f:
        f.write("\n".join(ratings.collect()))
    end = time.time()
    #print("Duration: ", end - start)

def get_user_based_rating(u_id, b_id, bu_map,ub_map, user_sim, avg_user_dict,avg_rating,rev_user_dict,users_dict):
    b_users = set(list(bu_map[b_id].keys())).union(set([u_id]))
    final_uid = rev_user_dict.get(u_id, "UNK")
    avg_user_rating = {}
    for user in b_users:
        avg_user_rating[user] = (avg_user_dict[user][0])/(avg_user_dict[user][1])
    b_users = b_users.difference(set([u_id]))
    num, denom = 0,0
    for user in b_users:
        sim = user_sim.get(tuple(sorted((user,u_id))),0)
        num += ((ub_map[user][b_id] - avg_user_rating.get(user,avg_rating))*sim)
        denom += abs(sim)
    return avg_user_rating.get(users_dict[final_uid],avg_rating) if denom == 0 else (avg_user_rating.get(users_dict[final_uid],avg_rating) + num/denom)

def get_item_based_rating(u_id,b_id,data,bus_sim, rev_b_dict, avg_bid_map,avg_rating):
    score_sim = []
    final_bid = rev_b_dict.get(b_id, "UNK")
    for bidx, score in data.items():
        score_sim.append((score,bus_sim.get(tuple(sorted((bidx,b_id))),0)))
    score_sim = sorted(score_sim, key = lambda x: x[1], reverse=True)

    num, denom = 0,0
    for i in score_sim:
        num += (i[0] * i[1])
        denom += abs(i[1])
    if denom == 0 or num == 0:
        return avg_bid_map.get(final_bid, avg_rating)
    return (num / denom)

if __name__ == "__main__":
    main()