import math
import heapq

file_name = "datasets/hac_datasets/data1.txt"
final_cluster_count = 2

dataset = []
clusters = {}
heap = []
dataset_size = 0
dimension = 2

def load_data(input_file_name):
    input_file = open(input_file_name, 'r')

    id = 0
    for line in input_file:
        row = line.strip('\n').split(" ")

        data = {}
        data.setdefault("id", id)
        data.setdefault("data", row[:])
        dataset.append(data)

        clusters.setdefault(id, {})
        clusters[id].setdefault("points", row[:])
        clusters[id].setdefault("elements", [id])

        id += 1
    return dataset, clusters

def hierarchical_clustering():
    current_clusters = clusters
    old_clusters = []
    heap = distance_between_points(dataset)
    heap = build_priority_queue(heap)

    while len(current_clusters) > final_cluster_count:
        dist, min_item = heapq.heappop(heap)

        # judge if include old cluster
        if not valid_heap_node(min_item, old_clusters):#node lar old_cluster icinde var mi diye bakiyor
            continue

        new_cluster = {}
        new_cluster_elements = sum(min_item, [])
        new_cluster_points = []
        for i in range(len(new_cluster_elements)):
            new_cluster_points.append(dataset[new_cluster_elements[i]]["data"])
        new_cluster_elements.sort()
        new_cluster.setdefault("points", new_cluster_points)
        new_cluster.setdefault("elements", new_cluster_elements)
        for pair_item in min_item:
            old_clusters.append(pair_item)
            del current_clusters[str(pair_item)]
        add_heap_entry(heap, new_cluster, current_clusters)
        current_clusters[str(new_cluster_elements)] = new_cluster
    current_clusters.sort()
    return current_clusters

def add_heap_entry(heap, new_cluster, current_clusters):
    for current_cluster in current_clusters.values():
        new_heap_entry = []
        dist = euclidean_distance_between_clusters(current_cluster["points"], new_cluster["points"])
        new_heap_entry.append(dist)
        new_heap_entry.append(sum[new_cluster["elements"],[]], current_cluster["elements"])
        heapq.heappush(heap, (dist, new_heap_entry))

def euclidean_distance(new_point, current_point):
    result = ((float(new_point[0]) - float(current_point[0]))**2) + ((float(new_point[1]) - float(current_point[1]))**2)
    result = math.sqrt(result)
    return result

def distance_between_points(dataset):
    result = []
    dataset_size = len(dataset)
    for i in range(dataset_size - 1):
        for j in range(i + 1, dataset_size):
            dist = euclidean_distance(dataset[i]["data"], dataset[j]["data"])
            result.append((dist, [[str(i)], [str(j)]]))
    return result

def build_priority_queue(distance_list):
    heapq.heapify(distance_list)
    heap = distance_list
    return heap

def valid_heap_node(heap_node, old_clusters):
    pair_data = heap_node
    for old_cluster in old_clusters:
        if old_cluster in pair_data:
            return False
    return True

def euclidean_distance_between_clusters(current_point, new_points):
    result = 9999999999999999
    for point in new_points:
        tmp = ((float(point[0]) - float(current_point[0]))**2) + ((float(point[1]) - float(current_point[1]))**2)
        if tmp < result:
            result = tmp
    result = math.sqrt(result)
    return result

def display(current_clusters):
    clusters = current_clusters.values()
    for cluster in clusters:
        cluster["elements"].sort()
        print(cluster["elements"])

if __name__ == '__main__':
    try:
        dataset, clusters = load_data(file_name)
        current_clusters = hierarchical_clustering()
        display(current_clusters)
    except Exception as e:
        print("error: " + str(e))