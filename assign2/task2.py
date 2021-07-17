from pyspark.sql import SparkSession
import sys,csv,math,time
from collections import defaultdict

def main():
    start = time.time()
    spark = SparkSession.builder.appName("assign1_task1").master("local[*]").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")
    sc.setSystemProperty('spark.driver.memory', '4g')
    sc.setSystemProperty('spark.executor.memory', '4g')
    
    args = sys.argv
    filter_threshold, support, inp_file, out_file = int(args[1]),int(args[2]),str(args[3]),str(args[4])

    input_rdd = sc.textFile(inp_file).map(lambda x : x.split(","))
    son(input_rdd,support, out_file, filter_threshold)
    end = time.time()
    print("Duration: ",round(end - start,2))

def son(input_rdd,support, out_file, filter_threshold):
    header = input_rdd.first()
    baskets = input_rdd.filter(lambda x: x != header).map(lambda x : (x[0],frozenset( [x[1]] ) )).reduceByKey(lambda x,y : x | y)
    baskets = baskets.filter(lambda x:len(x[1]) > filter_threshold).map(lambda x: x[1])
    total_items = baskets.count()
    candidate_itemsets = baskets.mapPartitions(lambda x: apriori(x,support,total_items)).groupByKey().map(lambda x: x[0]).collect()

    frequent_itemsets = baskets.mapPartitions(lambda x : get_item_count(x,candidate_itemsets)).reduceByKey(lambda x,y: x + y).filter(lambda x: x[1] >= support).map(lambda x :list(x[0]))
    frequent_itemsets = frequent_itemsets.collect()

    display_format(candidate_itemsets,frequent_itemsets, out_file)

def display_format(candidate_itemsets,frequent_itemsets, out_file):
    f = open(out_file,"w+")
    f.write("Candidates:\n")
    format_data(candidate_itemsets,f)
    f.write("Frequent Itemsets:\n")
    format_data(frequent_itemsets,f)
    f.close()

def format_data(itemset,f):
    length_map = defaultdict(lambda : [])
    for i in itemset:
        length_map[len(i)].append(sorted(tuple(i)))
    for items in length_map.values():
        items.sort()
        f.write(",".join(list(map(lambda x: str(tuple(x)).replace(',)',')'),items))))
        f.write("\n\n")

def get_item_count(input_data,candidate_itemsets):
    counts_map = defaultdict(int)
    input_data = list(input_data)
    for item in candidate_itemsets:
        for data in input_data:
            if item.issubset(data):
                counts_map[item] += 1
    return counts_map.items()

def get_candidate_items(frequent_items,k):
    candidate_items = set()
    for i in range(len(frequent_items)-1):
        for j in range(i+1,len(frequent_items)):
            candidate = frequent_items[i].union(frequent_items[j])
            if len(candidate) == k:
                candidate_items.add(candidate)
    return candidate_items

def get_frequent_items(candidate_items, baskets_arr,partial_support, all_frequent_items):
    counts_map = defaultdict(int)
    for i in candidate_items:
        for j in baskets_arr:
            if i.issubset(j):
                counts_map[i] += 1
    
    frequent_items = []
    for key,value in counts_map.items(): 
        if value >= partial_support:
            frequent_items.append(key)
            all_frequent_items.append((key,1))
    return frequent_items,all_frequent_items


def apriori(baskets, support, total_items):
    baskets_arr = [] 
    all_frequent_items = []
    counts_map = defaultdict(int)
    frequent_items = []
    for basket in baskets:
        baskets_arr.append(basket)
        for item in basket:
            counts_map[ frozenset( (item,) ) ] += 1

    partial_support = support * (len(baskets_arr) / total_items)
    
    for key,value in counts_map.items():
        if value >= partial_support:
            all_frequent_items.append((key,1))
            frequent_items.append(key)

    k = 2
    while True:
        candidate_items = get_candidate_items(frequent_items,k)
        frequent_items, all_frequent_items = get_frequent_items(candidate_items, baskets_arr,partial_support, all_frequent_items)

        if len(candidate_items) == 0 or len(frequent_items) == 0:
            break
        k += 1
    return all_frequent_items


if __name__ == "__main__":
    main()
