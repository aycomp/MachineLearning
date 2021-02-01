import glob

file_path = "datasets/hac_datasets/data1.txt"
point_list = []
df_min_distance = 999999999999
class data_point(object):
    def __init__(self, index=None, x=None, y=None, min_distance=None, neighbor_index=None):
        self.index = index
        self.x = x
        self.y = y
        self.min_distance = min_distance
        self.neighbor_index = neighbor_index

# load txt file into data_point_list which consists of data_point objects
def load_data(file):
    counter = 0
    with open(file, 'r') as f:
        for line in f:
            line = line.rstrip().split()
            point = data_point()
            point.index = counter
            point.x = line[0]
            point.y = line[1]
            point.min_distance = df_min_distance
            point_list.append(point)
            counter = counter + 1
    return point_list

def minDistance(point):
    return point.min_distance

def doOperations():
    cluster_count = len(point_list)
    centroid_counter = len(point_list)
    while cluster_count > 2:
        for i in range(len(point_list)):
            min_distance = df_min_distance
            tmp_neighbor_index = 0
            for j in range(i+1,len(point_list)):
                tmp = ((float(point_list[i].x) - float(point_list[j].x))**2 + (float(point_list[i].y) - float(point_list[j].y))**2)** 1/2
                if tmp < min_distance:
                    min_distance = tmp
                    tmp_neighbor_index = j
            if(min_distance != df_min_distance):
                point_list[i].min_distance = min_distance
                point_list[i].neighbor_index = tmp_neighbor_index

        try:
            point_list.sort(key=minDistance)
        except Exception as e:
            print(str(e))


        #centroids to delete
        del1 = point_list[0]
        del2 = [item for item in point_list if item.index == del1.neighbor_index]

        if(len(del2) > 0):
            del2 = del2[0]

            #centroids to add
            point = data_point()
            point.index = centroid_counter
            point.x = ((float)(del1.x) + (float)(del2.x)) / 2
            point.y = ((float)(del1.y) + (float)(del2.y)) / 2
            point.min_distance = df_min_distance
            point_list.append(point)

            for i, o in enumerate(point_list):
                if o.index == del1.index:
                    del point_list[i]
                    print(str(point_list[i].index) + " ######" + str(point_list[i].x) + "#" + str(point_list[i].y))
                    break

            for i, o in enumerate(point_list):
                if o.index == del2.index:
                    del point_list[i]
                    print(str(point_list[i].index) + "######" + str(point_list[i].x) + "#" + str(point_list[i].y))
                    break
        else:
            for i, o in enumerate(point_list):
                if o.index == del1.index:
                    del point_list[i]
                    print(str(point_list[i].index) + " ######" + str(point_list[i].x) + "#" + str(point_list[i].y))
                    break

        cluster_count = cluster_count - 1
        centroid_counter = centroid_counter + 1

if __name__ == '__main__':
    print('program started...')
    files = glob.glob(file_path)
    files.sort()

    for file in files:
        load_data(file)
        break
    x = []
    y = []

    try:
        for item in point_list:
            x.append(item.x)
            y.append(item.y)
    except Exception as e:
        print(e)


    print(len(point_list))
    print('program finished...')