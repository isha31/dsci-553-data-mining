from pyspark.sql import SparkSession
import json, time, sys, math, random
from itertools import combinations

CORATE = 3
HASH_FUNC_NO = 30
BANDS = 30
ROWS = 1
SIMILARITY_THRESHOLD = 0.01

def main():
    start = time.time()
    spark = SparkSession.builder.appName("assign3_task1").master("local[*]").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")
    sc.setSystemProperty('spark.driver.memory', '4g')
    sc.setSystemProperty('spark.executor.memory', '4g')

    args = sys.argv
    train_file, model_file, cf_type = str(args[1]), str(args[2]), str(args[3])
    input_rdd = spark.read.json(train_file).rdd.map(lambda x: (x["business_id"],x["user_id"],x["stars"]))

    bus_idx = input_rdd.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
    rev_bus_idx = {v:k for k,v in bus_idx.items()}

    user_idx = input_rdd.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
    reverse_user_index = {v:k for k,v in user_idx.items()}
    similar_pairs = []

    if cf_type == "item_based":
        business_users = input_rdd.map(lambda x: (bus_idx[x[0]],(user_idx[x[1]],x[2]))) \
            .groupByKey() \
            .mapValues(lambda x: list(x)) \
            .filter(lambda x: len(x[1]) >= CORATE) \
            .map(lambda x : (x[0], dict(x[1])))
        business_users_map = business_users.collectAsMap()
        business_users = business_users.map(lambda x:x[0])

        candidate_business_pairs = business_users.cartesian(business_users) \
            .filter(lambda x: x[0] < x[1])

        similar_pairs = candidate_business_pairs\
            .filter(lambda x: filter_corated_pairs(x, business_users_map)) \
            .map(lambda x:(rev_bus_idx[x[0]], rev_bus_idx[x[1]],get_pearson_corelation_item(x[0],x[1], business_users_map))) \
            .filter(lambda x : x[2] > 0) \
            .map(lambda x : "{"+f'"b1": "{x[0]}", "b2": "{x[1]}", "sim": {x[2]}'+"}")

    elif cf_type == "user_based":
        ubs_map = input_rdd.map(lambda x: (user_idx[x[1]],(bus_idx[x[0]] ,x[2]))) \
            .groupByKey() \
            .mapValues(lambda x: list(x)).mapValues(lambda x: dict(x))

        hash_functions = get_hash_functions(len(user_idx))
        ub_data = ubs_map.map(lambda x: (x[0], set([bus for bus, score in x[1].items()]))).collectAsMap()

        signature_matrix = ubs_map.map(lambda row: (row[0],get_signature_matrix(row[1].keys(), hash_functions)))
        similar_pairs = signature_matrix.flatMap(lambda row: apply_lsh(row)) \
            .groupByKey() \
            .map(lambda x : (sorted(set(x[1])))) \
            .filter(lambda val: len(val) > 1) \
            .flatMap(lambda row : get_similar_pairs(row,ub_data)).distinct()

        ubs_map = ubs_map.collectAsMap()
        similar_pairs = similar_pairs.map(lambda x : (reverse_user_index[x[0][0]],reverse_user_index[x[0][1]],get_pearson_corelation_user(x[0][0], x[0][1], ubs_map))) \
            .filter(lambda x: x[2] > 0) \
            .map(lambda x : "{"+f'"u1": "{x[0]}", "u2": "{x[1]}", "sim": {x[2]}'+"}") \


    with open(model_file, 'w+',encoding='utf-8') as f:
        f.write("\n".join(similar_pairs.collect()))
    end = time.time()
    print("DURATION: ", end-start)

def get_hash_functions(users_count):
    hash_functions = []
    for i in range(HASH_FUNC_NO):
        hash_functions.append(get_hash_function(users_count))
    return hash_functions

def get_hash_function(users_count):
    a = random.randint(2, 10000000000)
    b = random.randint(2, 10000000000)
    m = 999999991  #large prime number
    def hash_func(x):
        return ((a * x + b) % m)%users_count
    return hash_func

def get_similar_pairs(row,user_business_data):
    similar_pairs = {}
    candidate_pairs = set(combinations(row, 2))
    for candidate in candidate_pairs:
        if len(user_business_data[candidate[0]].intersection(user_business_data[candidate[1]])) >= CORATE:
            similarity = get_jacard_similarity(candidate, user_business_data)
            if similarity >= SIMILARITY_THRESHOLD:
                similar_pairs[candidate] = similarity
    return similar_pairs.items()

def get_jacard_similarity(pair, business_user_map):
    set1 = business_user_map.get(pair[0])
    set2 = business_user_map.get(pair[1])
    return len(set1.intersection(set2)) / len(set1.union(set2))


def get_signature_matrix(row, hash_functions):
    hash_arr = [float('inf') for _ in range(HASH_FUNC_NO)]
    for idx in row:
        for h in range(HASH_FUNC_NO):
            hash_arr[h] = min(hash_functions[h](idx), hash_arr[h])
    return hash_arr

def apply_lsh(row):
    lsh_buckets = []
    for i in range(BANDS):
        bucket = row[1][i*ROWS:i*ROWS + ROWS]
        lsh_buckets.append(((tuple(bucket), i), row[0]))
    return lsh_buckets

def get_pearson_corelation_item(x1,x2, score_map):
    corated_items = set(score_map[x1].keys()) & set(score_map[x2].keys())
    avg1, avg2 = 0,0
    for i in corated_items:
        avg1 += score_map[x1][i]
        avg2 += score_map[x2][i]
    avg1 = avg1 / len(corated_items)
    avg2 = avg2 / len(corated_items)
    num, denom_x1, denom_x2 = 0,0,0
    for i in corated_items:
        num += (score_map[x1][i] - avg1) * (score_map[x2][i] - avg2)
        denom_x1 += (score_map[x1][i] - avg1) ** 2
        denom_x2 += (score_map[x2][i] - avg2) ** 2
    denom = math.sqrt(denom_x1) * math.sqrt(denom_x2)
    pearson_coeff = 0 if denom == 0 else num / denom
    return  pearson_coeff

def get_pearson_corelation_user(u1, u2, score_map):
    corated_items = set(score_map[u1].keys()) & set(score_map[u2].keys())

    b_stars1 = { k:v for k,v in score_map[u1].items() if k in corated_items }
    b_stars2 = { k:v for k,v in score_map[u2].items() if k in corated_items }

    avg1 = sum(b_stars1.values())/len(b_stars1)
    avg2 = sum(b_stars2.values())/len(b_stars2)

    for k,v in b_stars1.items(): b_stars1[k] = v-avg1
    for k,v in b_stars2.items(): b_stars2[k] = v-avg2

    num, denom1, denom2 = 0, 0, 0
    for bus in corated_items:
        num += b_stars1[bus] * b_stars2[bus]
        denom1 += b_stars1[bus]**2
        denom2 += b_stars2[bus]**2

    denom = (math.sqrt(denom1)*math.sqrt(denom2))
    if denom==0: return 0
    else: return num/denom

def filter_corated_pairs(x, business_users_score):
    if x[0] in business_users_score and x[1] in business_users_score:
        return len(set(business_users_score[x[0]].keys()).intersection(set(business_users_score[x[1]].keys()))) >= CORATE


if __name__ == "__main__":
    main()