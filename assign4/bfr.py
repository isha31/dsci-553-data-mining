import sys,os,random,math,time, csv,json
from collections import OrderedDict, defaultdict
from pyspark.sql import SparkSession
from itertools import combinations
from functools import reduce

MAX_ITERATIONS = 20
RS_THRESHOLD = 1
ALPHA = 3
HEADER = ["round_id","nof_cluster_discard","nof_point_discard","nof_cluster_compression","nof_point_compression","nof_point_retained"]

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = 'usr/local/bin/python3.6'

spark = SparkSession.builder.appName("assign3_task1").master("local[*]").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("OFF")
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')


def main():
    start = time.time()
    args = sys.argv
    input_path = str(args[1])
    n_clusters = int(args[2])
    out_file1 = str(args[3])
    out_file2 = str(args[4])

    test_files = sorted(list(os.listdir(input_path)))
    
    for round_id,input_file in enumerate(test_files):
        input_rdd = sc.textFile(os.path.join(input_path,input_file)) \
            .map(lambda x: x.split()) \
            .map(lambda x: x[0].split(",")) \
            .map(lambda x: (int(x[0]),list(map(float,x[1:]))))
        if round_id == 0: 
            subset_data1 = input_rdd.sample(withReplacement=False,fraction=0.1)
            subset_data1_keys = set(subset_data1.map(lambda x: x[0]).collect())
            subset_data2 = input_rdd.filter(lambda x: x[0] not in subset_data1_keys)

            subset_map = subset_data1.collectAsMap()
            dim = len(list(subset_map.values())[0])  

            res = kmeans(subset_map,n_clusters*3,dim,MAX_ITERATIONS)
            inlier,outlier = {},{}
            for k,v in res.items():
                if (len(v["points"]) > RS_THRESHOLD):
                    inlier = {**inlier,**dict(v["points"])}
                else:
                    outlier = {**outlier,**dict(v["points"])}

            discard_set = kmeans(inlier,n_clusters,dim,MAX_ITERATIONS) if len(inlier) > 0 else {}
            discard_set = discard_points(discard_set)

            compressed_set, retained_set = {},[]
            if len(outlier) > 0: 
                res = kmeans(outlier,n_clusters * 3,dim,MAX_ITERATIONS)
                for k,v in res.items():
                    if (len(v["points"]) > RS_THRESHOLD):
                        compressed_set[k] = v
                    else:
                        retained_set += v["points"]
            compressed_set = discard_points(compressed_set)

        if round_id > 0: subset_data2 = input_rdd
        subset_data2 = subset_data2.map(lambda x :((mahalanobis_cluster(x[1],discard_set),(x[0],x[1])))) \
            .groupByKey() \
            .mapValues(lambda x: list(x))

        discard_set2 = subset_data2.filter(lambda x: x[0] != -1).map(lambda cluster : (cluster[0] ,merge_dset(cluster[1], discard_set[cluster[0]],dim))).collectAsMap()
        discard_set = {**discard_set, **discard_set2}

        rem_points = subset_data2.filter(lambda x: x[0] == -1) \
            .flatMap(lambda x: x[1]) \
            .map(lambda x: ((mahalanobis_cluster(x[1],compressed_set),(x[0],x[1])))) \
            .groupByKey() \
            .mapValues(lambda x: list(x))

        compressed_set2 = rem_points.filter(lambda x: x[0] != -1).map(lambda cluster : (cluster[0] ,merge_dset(cluster[1], compressed_set[cluster[0]],dim))).collectAsMap()
        compressed_set2 = {**compressed_set, **compressed_set2}

        retained_set2 = rem_points.filter(lambda x: x[0] == -1).flatMap(lambda x: x[1]).collectAsMap()
        retained_set2 = {**dict(retained_set), ** retained_set2}

        compressed_set3, retained_set = {},[]
        if len(retained_set2) > 0:
            res = kmeans(retained_set2,n_clusters*3,dim,MAX_ITERATIONS)
            for k,v in res.items():
                if (len(v["points"]) > RS_THRESHOLD):
                    compressed_set3[k] = v
                else:
                    retained_set += v["points"]

        compressed_set3 = discard_points(compressed_set3)
        compressed_set3 = dict(enumerate(list(compressed_set3.values()) + list(compressed_set2.values())))
        compressed_set = merge_cs_clusters(compressed_set3,dim)
        write_intermediate_file(out_file2,discard_set,compressed_set,retained_set, round_id)

    discard_set = merge_cs_ds(compressed_set, discard_set,dim)
    write_output_file(out_file1,discard_set,retained_set)

    end = time.time()
    #print("Duration: ", end-start)


def merge_cs_ds(cs, ds,dim):
    for cs_id, cs_data in cs.items():
        min_dist = float('inf')
        min_ds = None
        for ds_id, ds_data in ds.items():
            dist = mhlb_dist(cs_data["centroid"],ds_data["centroid"],ds_data["std"])
            if dist < min_dist:
                min_dist = dist
                min_ds = ds_id
        ds[min_ds] = merge_cluster(ds[min_ds], cs_data, dim)
    return ds

def write_output_file(out_file1,discard_set,retained_set):
    data_map = {}
    for cluster, data in discard_set.items():
        for point in data["points"]:
            data_map[point] = int(cluster[1:])

    for point in retained_set:
        data_map[point[0]] = -1

    data_map = OrderedDict(sorted(data_map.items()))
    
    with open(out_file1, "w+") as json_file:
        json_file.write(json.dumps(data_map))

def write_intermediate_file(out_file,ds,cs,rs,round_id):
    mode = 'a+'
    if round_id == 0:mode = 'w+'

    with open(out_file,mode) as csv_file:
        csv_writer = csv.writer(csv_file)
        if round_id == 0:
            csv_writer.writerow(HEADER)
        ds_count = reduce(lambda x, value:x + value["n"], ds.values(), 0)
        cs_count = reduce(lambda x, value:x + value["n"], cs.values(), 0)
        csv_writer.writerow([round_id+1,len(ds), ds_count, len(cs), cs_count, len(rs)])

def merge_cs_clusters(compressed_set, dim):
    for cl1, cl2 in combinations(compressed_set.keys(),2):
        if cl1 in compressed_set and cl2 in compressed_set:
            dist = mhlb_dist(compressed_set[cl2]["centroid"],compressed_set[cl1]["centroid"],compressed_set[cl1]["std"])
            if dist < ALPHA * math.sqrt(dim):
                compressed_set[cl1] = merge_cluster(compressed_set[cl1],compressed_set[cl2],dim)
                del compressed_set[cl2]
    return compressed_set

def merge_cluster(cluster1, cluster2, dim):
    for i in range(dim):
        cluster1["sum"][i] += cluster2["sum"][i]
        cluster1["sumsq"][i] += cluster2["sumsq"][i]
    cluster1["n"] += cluster2["n"]
    cluster1["points"] += cluster2["points"]
    cluster1["std"] = compute_std(cluster1,dim)
    cluster1["centroid"] = compute_centroids(cluster1)
    return cluster1

def merge_dset(data_points, cluster_data, dim):
    for point in data_points:
        for id,val in enumerate(point[1]):
            cluster_data["sum"][id] += val
            cluster_data["sumsq"][id] += val**2
        cluster_data["n"] += 1
        cluster_data["points"].append(point[0])
    cluster_data["std"] = compute_std(cluster_data, dim)
    cluster_data["centroid"] = compute_centroids(cluster_data)
    return cluster_data

def compute_centroids(cluster_data):
    return [i/cluster_data["n"] for i in cluster_data["sum"]]

def compute_std(cluster_data,dim):
    std_data = [math.sqrt((cluster_data["sumsq"][i]/cluster_data["n"])-((cluster_data["sum"][i]/cluster_data["n"])**2)) for i in range(dim)]
    return std_data

def discard_points(cluster_set):
    for k,v in cluster_set.items(): v["points"] = list(dict(v["points"]).keys())
    return cluster_set


def mahalanobis_cluster(point, discard_set):
    min_dist = float('inf')
    min_cluster = None
    d_sqrt = math.sqrt(len(point))
    
    for cluster, values in discard_set.items():
        dist = mhlb_dist(point,values["centroid"],values["std"])
        if dist < min_dist :
            min_dist = dist
            min_cluster = cluster
    return min_cluster if (min_dist < (ALPHA * d_sqrt)) else -1

def mhlb_dist(cluster1, cluster2, std):
    sum_val = 0
    for (c1,c2,sd) in zip(cluster1,cluster2,std):
        if sd == 0:
            sum_val += 0
        else:
            sum_val += ((c1 - c2) / sd) ** 2
    return float(math.sqrt(sum_val))

def kmeans(datapoints,k,dim, max_iterations):
    datapoints_rdd = sc.parallelize(datapoints.items())
    centroids = initial_k_centroids(datapoints,k)
    prev_info = centroids
    
    iter_count = 0
    while True:
        centroid_data = datapoints_rdd.map(lambda x: (get_cluster(x,centroids),(x[0],x[1]))) \
            .groupByKey() \
            .map(lambda cluster: (cluster[0], (new_centroids(list(cluster[1]), dim)))) 
        centroids = centroid_data.map(lambda x: (x[0],x[1]["centroid"])).collectAsMap()
        centroids = {**prev_info, **centroids}
        if compare_centroids(prev_info,centroids) or iter_count >= max_iterations:
            break
        prev_info = centroids
        iter_count += 1
    centroid_data = centroid_data.collectAsMap()

    centroid_map = defaultdict(dict)
    for c,val in centroids.items():
        centroid_map[c]["points"] = []
        centroid_map[c]["n"] = 0
        centroid_map[c]["centroid"] = val
    centroid_data = {**centroid_map, **centroid_data}
    return centroid_data

def compare_centroids(prev_info, curr_info):
    for cluster_id,dim in prev_info.items():
        if dim != curr_info[cluster_id]:
            return False
    return True

def new_centroids(cluster,dim):
    cluster_len = len(cluster) 
    cluster_dim = [0] * dim
    cluster_sq_dim = [0] * dim
    for point in cluster:
        for id,val in enumerate(point[1]):
            cluster_dim[id] += val
            cluster_sq_dim[id] += (val ** 2)
    cluster_data = {"std":[],"n":cluster_len,"sum":cluster_dim,"sumsq":cluster_sq_dim,"points":cluster}
    cluster_data["centroid"] = compute_centroids(cluster_data)
    cluster_data["std"] = compute_std(cluster_data,dim)
    return cluster_data

def initial_k_centroids(points,k):
    # if len(points) >= k:
    #     rkeys = random.sample(points.keys(),k)
    # else:
    #     rkeys = points.keys()
    # return {f'c{id}':points[p] for id,p in enumerate(rkeys)}
    
    k_points = {}
    init_pnt = random.choice(list(points.keys()))
    k_points[init_pnt] = points[init_pnt]
    del points[init_pnt]

    for i in range(0,k-1):
        max_dist = -float('inf')
        max_val = None
        rand_pnts = random.sample(list(points.keys()),10)
        for pnt in rand_pnts:
            dist = math.sqrt(sum([(k_points[init_pnt][i]-val)**2 for i,val in enumerate(points[pnt])]))
            if dist > max_dist:
                max_dist = dist
                max_val = pnt
        init_pnt = max_val
        k_points[init_pnt] = points[init_pnt]
        del points[init_pnt]

    return k_points
        





    # if len(points) <= 0:
    #     return k_points
    # curr = random.choice(list(points.keys()))
    # curr_label = "c0"
    # k_points[curr_label] =  points[curr]
    # del points[curr]

    # for i in range(0,k-1):
    #     max_dist = -float('inf')
    #     max_point = None
    #     for point, dim in points.items():
    #         dist = math.sqrt(sum([(k_points[curr_label][i]-val)**2 for i,val in enumerate(dim)]))
    #         if dist > max_dist:
    #             max_dist = dist
    #             max_point = (point, dim)
    #     curr_label = f'c{i+1}'
    #     if max_point:
    #         curr = max_point[0]
    #         k_points[curr_label] = max_point[1]
    #         del points[curr]

    # return k_points

def get_cluster(point,centroids):
    min_dist = float('inf')
    cluster = None
    for cluster_idx, dim in centroids.items():
        dist = math.sqrt(sum([(point[1][i]-val)**2 for i,val in enumerate(dim)]))
        if dist < min_dist:
            min_dist = dist
            cluster = cluster_idx
    return cluster

if __name__ == "__main__":
    main()