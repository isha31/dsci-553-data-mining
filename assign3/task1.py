from pyspark.sql import SparkSession
import sys,time, random
from collections import defaultdict
from itertools import combinations

HASH_FUNC_NO = 50
BANDS = 50
ROWS = 1
SIMILARITY_THRESHOLD = 0.05

def main():
    start = time.time()
    spark = SparkSession.builder.appName("assign3_task1").master("local[*]").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")
    sc.setSystemProperty('spark.driver.memory', '4g')
    sc.setSystemProperty('spark.executor.memory', '4g')

    args = sys.argv
    inp_file,out_file = str(args[1]),str(args[2])

    reviews_rdd = spark.read.json(inp_file).rdd
    users = reviews_rdd.map(lambda x: x["user_id"]).distinct().collect()
    users_index = {user:idx for idx, user in enumerate(users)}
    users_count = len(users_index)

    business = reviews_rdd.map(lambda x: x["business_id"]).distinct().collect()
    business_index = {business:idx for idx, business in enumerate(business)}
    reverse_business_index = {val:key for key, val in business_index.items()}
    hash_functions = get_hash_functions(users_count)

    business_user_map =  reviews_rdd.map(lambda x:(business_index[x["business_id"]],{users_index[x["user_id"]]})) \
        .reduceByKey(lambda x,y: x | y)
    business_user_data = business_user_map.collectAsMap()
    signature_matrix = business_user_map.map(lambda row: (row[0], get_signature_matrix(row, hash_functions)))
    similar_pairs = signature_matrix \
        .flatMap(lambda row: apply_lsh(row)) \
        .groupByKey() \
        .map(lambda x : (x[0], sorted(set(x[1])))) \
        .flatMap(lambda row : get_similar_pairs(row,business_user_data)) \
        .collectAsMap()

    f = open(out_file, "w+")
    for pair in similar_pairs:
        f.write("{"+"\"b1\": \"{}\",\"b2\": \"{}\", \"sim\": {}".format(reverse_business_index[pair[0]], reverse_business_index[pair[1]],similar_pairs[pair]) + "}\n")
    f.close()

    end = time.time()

def apply_lsh(row):
    lsh_buckets = []
    for i in range(BANDS):
        bucket = row[1][i*ROWS:i*ROWS + ROWS]
        lsh_buckets.append(((tuple(bucket), i), row[0]))
    return lsh_buckets

def get_similar_pairs(row,business_user_data):
    similar_pairs = {}
    candidate_pairs = set(combinations(row[1], 2))
    for candidate in candidate_pairs:
       similarity = get_jacard_similarity(candidate, business_user_data)
       if similarity >= SIMILARITY_THRESHOLD:
           similar_pairs[candidate] = similarity
    return similar_pairs.items()

def get_jacard_similarity(pair, business_user_map):
    set1 = business_user_map.get(pair[0])
    set2 = business_user_map.get(pair[1])
    return len(set1.intersection(set2)) / len(set1.union(set2))


def get_signature_matrix(row, hash_functions):
    hash_arr = [float('inf') for _ in range(HASH_FUNC_NO)]
    for user_idx in row[1]:
        for h in range(HASH_FUNC_NO):
            hash_func = hash_functions[h]
            hash_val = hash_func(user_idx)
            hash_arr[h] = min(hash_val, hash_arr[h])
    return hash_arr

def get_hash_functions(users_count):
    hash_functions = []
    for i in range(HASH_FUNC_NO):
        hash_functions.append(get_hash_function(users_count)) 
    return hash_functions

def get_hash_function(users_count):
    a = random.randint(2, 10000000000)
    b = random.randint(2, 10000000000)
    m = 999999991
    def hash_func(x):
        return ((a * x + b) % m)%users_count
    return hash_func

if __name__ == "__main__":
    main()