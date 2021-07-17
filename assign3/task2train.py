from pyspark.sql import SparkSession
import sys, time, json, re, math
from collections import defaultdict

RARE_WORDS_FREQ = 0.000001

def main():
    start = time.time()
    spark = SparkSession.builder.appName("assign3_task1").master("local[*]").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")
    sc.setSystemProperty('spark.driver.memory', '4g')
    sc.setSystemProperty('spark.executor.memory', '4g')

    args = sys.argv
    train_file,model_file,stopwords = str(args[1]),str(args[2]),str(args[3])

    stopwords = set(sc.textFile(stopwords).collect())

    input_rdd = spark.read.json(train_file).rdd
    business = input_rdd.map(lambda x: x["business_id"]).distinct().collect()
    business_index = {business:idx for idx, business in enumerate(business)}
    reverse_business_index = {v:k for k,v in business_index.items()}
    documents_count = len(business)

    training_rdd = input_rdd.map(lambda x: (x["business_id"], re.sub(r'[\W+ \s+ \d+]', ' ', x["text"]).lower().split())).reduceByKey(lambda x,y: x + y)
    training_rdd = training_rdd.map(lambda x:(x[0], [word for word in x[1] if word not in stopwords]))

    word_count = training_rdd.flatMap(lambda x: [(w,1) for w in x[1]]).reduceByKey(lambda x,y : x + y)
    total_no_of_words = word_count.map(lambda x: x[1]).reduce(lambda x,y: x+ y)
    frequent_words = set(word_count.filter(lambda x: x[1] >= total_no_of_words * RARE_WORDS_FREQ).map(lambda x: x[0]).collect())
    training_rdd = training_rdd.map(lambda x: (x[0],[word for word in x[1] if word in frequent_words]))

    #TF - IDF
    tf_rdd = training_rdd.flatMap(lambda x: get_term_freq(x, business_index))
    idf_rdd = training_rdd.flatMap(lambda x: list(set([ (word,1) for word in x[1] ])) ).reduceByKey(lambda x,y: x + y).map(lambda x: (x[0],math.log(documents_count/x[1],2))).collectAsMap()
    tf_idf = tf_rdd.map(lambda x: (x[0],(x[1],x[2]*idf_rdd[x[1]]))).groupByKey().map(lambda x: (x[0],sorted(list(x[1]), key = lambda x: x[1], reverse=True)[:200]))

    # business profile
    business_profile = tf_idf.mapValues(lambda x: set([word[0] for word in x]))
    word_index_map = business_profile.flatMap(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
    business_profile = business_profile.mapValues(lambda x: [word_index_map[word] for word in x] )
    business_words = business_profile.collectAsMap()

    # user profile
    user_index = input_rdd.map(lambda x: x["user_id"]).distinct().zipWithIndex().collectAsMap()
    user_profile = input_rdd.map(lambda x: (x["user_id"],set((business_index[x["business_id"]],)))).reduceByKey(lambda x, y: x | y).map(lambda x: get_user_profile(x, business_words, reverse_business_index)) \
        .map(lambda x: "{"+f'"Type": "User", "ID": "{x[0]}", "Features": {x[1]}'+"}")
    business_profile = business_profile.map(lambda x: "{"+f'"Type": "Business", "ID": "{x[0]}", "Features": {x[1]}'+"}")

    print(user_profile.count())
    print(user_profile.take(5))

    # with open(model_file, 'w+') as f:
    #     f.write("\n".join(business_profile.collect()))
    #     f.write("\n")
    # with open(model_file, 'w+') as f:
    #     f.write("\n".join(user_profile.collect()))
    #     f.write("\n")
    end = time.time()

def get_user_profile(x, business_words, reverse_business_index):
    words = set()
    for business in x[1]:
        words = words.union(business_words[reverse_business_index[business]])
    return [x[0],list(words)]

def get_term_freq(x, business_index):
    tf_arr = []
    word_count_map = defaultdict(int)
    max_word_count = 0
    for i in x[1]:
        word_count_map[i] += 1
        max_word_count = max(max_word_count,word_count_map[i])
    for word, count in word_count_map.items():
        word_tf = tuple((x[0],word, count/max_word_count))
        tf_arr.append(word_tf)
    return tf_arr

if __name__ == "__main__":
    main()
