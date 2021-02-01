rows = 3
cols = 2

A = []
while len(A) < rows:
    A.append([])
    while len(A[-1]) < cols:
        A[-1].append(0.0)

print(A)



def calculate_distance_and_plot():
    plot_graph(single_linkage_criterion())
    plot_graph(complete_linkage_criterion())
    plot_graph(average_linkage_criterion())
    plot_graph(centroid_criterion())

def single_linkage_criterion():


def complete_linkage_criterion():


def average_linkage_criterion():


def centroid_criterion():


def plot_graph():
    # merge part
    for item in point_list:
        if item.min_distance == point_list[0].min_distance:
            del1 = item.index
            del2_index = point_list[item.neighbor_index]
            del2 = point_list[del2_index.index].index
            #
            point = data_point()
            point.index = centroid_counter
            point.x = ((float)([item for item in point_list if item.index == del1][0].x) +
                       (float)([item for item in point_list if item.index == del2][0].x)) / 2
            point.y = ((float)([item for item in point_list if item.index == del1][0].y) +
                       (float)([item for item in point_list if item.index == del2][0].y)) / 2

            point_list.append(point)
            del point_list[del1]
            del point_list[del2]
            break
    cluster_count = cluster_count - 1
    centroid_counter = centroid_counter + 1