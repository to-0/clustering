import math
import random
import matplotlib.pyplot as plt
import numpy as np
import time
#import plotly.graph_objs as go
#import altair as alt
border = 10000
time_add_remove = 0
all_time = 0
time_medoid = 0
time_centers = 0
time_distances = 0
time_matrix_things = 0
time_calculate_distance = 0
iterations_avg = 0
DEBUG = False
DIVISIVE = False
colors = ["red", "green", "blue", "orange", "purple", "brown", "black", "pink", "silver",
            "darksalmon", "gold", "palegreen", "deepskyblue", "navy", "mediumpurple",
          "plum", "palevioletred", "darkgoldenrod", "olive", "lawngreen", "darkseagreen", "forestgreen",
          "turquise", "lightseagreen", "lightcyan", "darkslategray", "aqua", "cadetblue", "lightblue", "slategrey",
          "lavender", "midnightblue", "indigo", "darkorchild", "hotpink"]
distance_matrix = list()
number_of_points = 0

def create_distance_matrix(points):
    global distance_matrix
    n = len(points)
    distance_matrix = [[0.0 for x in range(n)] for y in range(n)]
    i = 0
    for p1 in points:
        j = 0
        for p2 in points:
            dist = calculate_distance(p1.x, p1.y, p2.x, p2.y)
            if dist == 0:
                dist = float("inf")
            distance_matrix[i][j]=dist
            j += 1
        i += 1


def calculate_distance_w_matrix(p1, p2):
    i = p1.index
    j = p2.index
    return distance_matrix[i][j]



class Point:
    def __init__(self, x, y, cluster):
        self.x = x
        self.y = y
        self.cluster = cluster

class Cluster:
    def __init__(self, cluster_points):
        self.cluster_points = cluster_points
        self.x_sum = 0
        self.y_sum = 0
        self.initialize_averages()
        #self.matrix = self.create_matrix()

    def initialize_averages(self):
        for point in self.cluster_points:
            self.x_sum += point.x
            self.y_sum += point.y


    def calculate_centroid(self):
        x = self.x_sum / len(self.cluster_points)
        y = self.y_sum / len(self.cluster_points)
        return [x, y]

    def calculate_medoid_with_list(self, l):
        index = l.index(min(l))
        return self.cluster_points[index]

    def calculate_medoid_test(self):
        min_dis = -1
        med = -1
        for p1 in self.cluster_points:
            dist = 0
            for p2 in self.cluster_points:
                dist += calculate_distance(p1.x, p1.y, p2.x, p2.y)
            if min_dis == -1 or dist < min_dis:
                min_dis = dist
                med = p1
        return med

    def calculate_medoid_matrix(self):
        min_dis = -1
        med = -1
        for p1 in self.cluster_points:
            dist = 0
            for p2 in self.cluster_points:
                if p1 == p2:
                    continue
                dist += calculate_distance_w_matrix(p1, p2)
                #dist += calculate_distance(p1.x, p1.y, p2.x, p2.y)
            if min_dis == -1 or dist < min_dis:
                min_dis = dist
                med = p1
        return med

    def update_centroid_remove_point(self, point):
        self.x_sum -= point.x
        self.y_sum -= point.y
        self.cluster_points.remove(point)

    def calculate_medoid(self):
        if len(self.cluster_points)==1:
            return self.cluster_points[0]
        # vrati cislo najmensieho riadku
        i = 0
        min_d = -1
        index_medoid = -1
        for i in range(len(self.matrix[0])):
            num = self.matrix[0][i]
            if min_d == -1 or num < min_d:
                min_d = num
                index_medoid = i

        # print(self.matrix)
        return self.cluster_points[index_medoid]

    def create_matrix(self):
        length = len(self.cluster_points)
        matrix = [[0 for x in range(length)] for y in range(length)]
        if length == 1:
            return matrix
        i = 0
        row = []
        for point in self.cluster_points:
            j = 0
            for point2 in self.cluster_points:
                dist = calculate_distance(point.y, point.y, point2.x, point2.y)
                matrix[i][j] = dist
                j += 1
            i+= 1
        return matrix


def generate_points():
    points = []
    # prvych 20
    used_x = []
    used_y = []
    for i in range(20):
        x = random.randint(-5000, 5000)
        y = random.randint(-5000, 5000)
        points.append(Point(x, y, 0))

    #create_first_20()
    #global points
    print(len(points))
    for i in range(number_of_points):
        index = random.randint(0, len(points)-1)
        #print("dlzka je ", len(points), index)
        p = points[index]
        bottom_border_x = -100
        upper_border_x = 100
        bottom_border_y = -100
        upper_border_y = 100
        if (p.x - 100) < -5000:
            bottom_border_x = -5000 - p.x
        if p.x + 100 > 5000:
            upper_border_x = 5000 - p.x

        if p.y - 100 < -5000:
            bottom_border_y = -5000 - p.y
        if p.y + 100 > 5000:
            upper_border_y = 5000 - p.y

        x_offset = random.randint(bottom_border_x, upper_border_x)
        y_offset = random.randint(bottom_border_y, upper_border_y)
        point = Point(p.x+x_offset, p.y+y_offset, 0)
        points.append(point)
    return points

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distance


# vyberiem nahodne k bodov ako k clustrov
def select_k_clusters(clusters, k, points):
    used = []
    for i in range(k):
        c_number = random.randint(0, len(points)-1)
        while c_number in used:
            c_number = random.randint(0, len(points)-1)
        used.append(c_number)
        cluster = Cluster([points[c_number]])
        clusters.append(cluster)
        points[c_number].cluster = cluster


def draw(clusters):
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.yticks(np.arange(-5000, 6000, 1000))
    plt.yticks(np.arange(-5000, 6000, 1000))
    #fig, axs = plt.subplots()  # a figure with a 2x2 grid of Axes
    #print(axs)
    i = 0
    for cluster in clusters:
        for point in cluster.cluster_points:
            color = colors[i % len(colors)]
            plt.plot(point.x, point.y, marker=".", markersize=2, color=color)
        i+= 1

    plt.show()
    # for point in points:
    #     #print("Col ",point.cluster)
    #     color = colors[point.cluster.color]
    #     plt.plot(point.x, point.y, marker=".", markersize=2, color=color)

    plt.show()

def compute_cluster_centers(cent_med, clusters):
    centers = []
    for cluster in clusters:
        if cent_med == 1:
            centroid = cluster.calculate_centroid()
            centers.append(centroid)
        else:
            #medoid = cluster.calculate_medoid_test()
            medoid = cluster.calculate_medoid_with_list()
            centers.append([medoid.x, medoid.y])
    return centers

def create_list_of_distances(clusters):
    distances_list = []
    for cluster in clusters:
        clust_distances = []
        for point in cluster.cluster_points:
            sum_dist = 0
            for p2 in cluster.cluster_points:
                sum_dist += calculate_distance(point.x, point.y, p2.x, p2.y)
            clust_distances.append(sum_dist)
        distances_list.append(clust_distances)
    return distances_list

def compute_centroids(clusters):
    centers = []
    for cluster in clusters:
        centroid = cluster.calculate_centroid()
        centers.append(centroid)
    return centers

def compute_all_medoids(clusters):
    centers = []
    for cluster in clusters:
        # medoid = cluster.calculate_medoid_with_list(distances_list)
        medoid = cluster.calculate_medoid_test()
        centers.append([medoid.x, medoid.y])
    return centers


def k_means(clusters, points, k, cent_med):
    # init step
    global time_centers
    select_k_clusters(clusters, k, points)
    #print(clusters)
    iterations = 0
    global time_distances
    while True:
        if DEBUG:
            print("Iteration", iterations)
        changes = 0
        #STEP 1
        start_centeres = time.time()
        #print("Idem ratat centroidy/medoidy")
        if cent_med == 1:
            cluster_centers = compute_centroids(clusters)
        else:
            cluster_centers = compute_all_medoids(clusters)
        end_centers = time.time()
        time_centers += end_centers - start_centeres
        #print("Doratal som")
        for point in points:
            old_cluster = point.cluster
            min_dist = -1
            min_cluster = -1
            #print("Point dalsi")
            for i in range(len(cluster_centers)):
                center = cluster_centers[i]
                start = time.time()
                dist = calculate_distance(point.x, point.y, center[0], center[1])
                end = time.time()
                time_distances += (end-start)
                #print("dist", dist)
                if min_cluster == -1 or dist < min_dist:
                    min_dist = dist
                    min_cluster = clusters[i]

            if old_cluster != min_cluster:
                changes += 1

            if old_cluster != 0 and len(old_cluster.cluster_points) == 1 and old_cluster != min_cluster:
                continue
            if old_cluster != 0:
                old_cluster.cluster_points.remove(point)
                old_cluster.x_sum -= point.x
                old_cluster.y_sum -= point.y
            min_cluster.cluster_points.append(point)
            min_cluster.x_sum += point.x
            min_cluster.y_sum += point.y
            point.cluster = min_cluster

        if changes == 0:
            break
        iterations += 1
    global iterations_avg
    if not DIVISIVE:
        print("Number of iterations ", iterations)
        iterations_avg += iterations


# najvacsiu sumu vzdialenosti od centroidu ma
def calculate_worst_cluster(clusters):
    worst = 0
    worst_cluster = None
    for cluster in clusters:
        centroid = cluster.calculate_centroid()
        dist = 0
        for point in cluster.cluster_points:
            dist += calculate_distance(point.x, point.y, centroid[0], centroid[1])
        if dist > worst:
            worst = dist
            worst_cluster = cluster
    return worst_cluster

def divisive_clustering(clusters, points, k):
    global DIVISIVE
    DIVISIVE = True
    # vytvorim si random 2 clustre uz ich rozdelim ako keby
    k_means(clusters, points, 2, 1)
    print("-"*30)
    length = len(clusters)
    while length != k:
        # najdem si najhorsi cluster ten budem delit
        worst = calculate_worst_cluster(clusters)
        clusters.remove(worst)
        # toto mi prida dalsie 2 clustre cize ten stary cluster najhorsi predtym zahodim
        k_means(clusters, worst.cluster_points, 2, 1)
        length = len(clusters)

def find_closest_couple():
    min_di = -1
    min_couple = []
    i = 0
    for row in distance_matrix:
        s = min(row)
        j = row.index(s)
        if min_di == -1 or s < min_di:
            min_di = s
            min_couple = [i, j]
        i += 1
    return min_couple

def update_matrix_after_removal(keep_index, delete_index, new_cluster, clusters):
    global distance_matrix
    # distance_matrix = np.array(distance_matrix).delete(distance_matrix, row_i, 0)
    # distance_matrix = np.array(distance_matrix).delete(distance_matrix, row_i, 1)
    # vymazem jeden riadok lebo jeden row ide prec
    #distance_matrix.remove(distance_matrix[row_i])
    del distance_matrix[delete_index]
    # # v kazdom stlpci vymazem 1 prvok
    for row in distance_matrix:
        #row.remove(row[row_i])
        del row[delete_index]

    centroid = new_cluster.calculate_centroid()
    # -1 lebo jeden riadok som uz vymazal
    i = 0
    new_index =keep_index
    # prepocitam si nove vzdialenosti
    for cluster in clusters:
        centroid2 = cluster.calculate_centroid()
        dist = calculate_distance(centroid[0], centroid[1], centroid2[0], centroid2[1])
        if dist == 0:
            dist = float("inf")
        distance_matrix[new_index][i] = dist
        distance_matrix[i][new_index] = dist
        i += 1


def agglomerative_clustering(clusters, points, k):
    i = 0
    global distance_matrix
    print("Creating distance matrix")
    create_distance_matrix(points)
    print("Done")
    # pre kazdy bod spravim cluster
    for point in points:
        clusters.append(Cluster([point]))
    length = len(clusters)
    iteration = 0
    while length != k:
        print("Iteration ", iteration)
        # najdem najblizsiu dvojicu
        closest = find_closest_couple()
        print("Closest ", closest)
        if closest[0] == closest[1]:
            print("Something went wrong")
            exit(1)
        # necham si ten prvy index a ten druhy zahodim
        new_cluster = clusters[closest[0]]
        second_cluster = clusters[closest[1]]
        # prekopirujem body z toho jedneho clustra co idem vymazat do druheho ktory mi zostane
        print("Going to copy points ", len(second_cluster.cluster_points))
        j = 0
        for point in second_cluster.cluster_points:
            new_cluster.cluster_points.append(point)
            point.cluster = new_cluster
            print("K je ", j)
            new_cluster.x_sum += point.x
            new_cluster.y_sum += point.y
            j += 1
        # vymazem jeden z tych clustrov zo zoznamu
       # print("Idem vymazat?")
        clusters.remove(second_cluster)
        del second_cluster
        # updatnem maticu vymazem 1 riadok a stlpec a prepocitam jeden riadok a stlpec
        print("Updating matrix")
        update_matrix_after_removal(closest[0], closest[1], new_cluster, clusters)
        iteration += 1
        length = len(clusters)
        print("Length ", length)


def evaluate(clusters, avg_distances):
    all_clusters = len(clusters)
    bad_clusters = 0
    distances = []
    for cluster in clusters:
        center = cluster.calculate_centroid()
        #center = calculate_center(cluster)
        #plt.plot(center[0], center[1], marker="+", markersize=5, color="black")
        dist = 0
        for point in cluster.cluster_points:
            dist += calculate_distance(point.x, point.y, center[0], center[1])
        average_distance = dist / len(cluster.cluster_points)
        if average_distance > 500:
            bad_clusters += 1
        distances.append(average_distance)
    avg_distances.append(distances)
    good_clusters = all_clusters - bad_clusters
    print("Success rate ", good_clusters/all_clusters*100)
    return good_clusters/all_clusters*100

def clean_points(points):
    for point in points:
        point.cluster = 0


def testing_function():
    print("1. Test kmeans centroid")
    print("2. Test kmeans medoid")
    print("3. Test divisive clustering with centroids")
    print("4. Test agglomerative clustering with centroids")
    a = int(input())
    k = int(input("Select number of clusters\n"))
    n = int(input("Select how many times you want to run each cluster function\n"))
    global all_time
    draw_flag = int(input("1. Draw everything (slow)\n2. Draw only the best and worst\n3. Draw nothing\n"))
    points_flag = int(input("1. Generate new points every time\n2. Use the same points (except agglomerative clustering)\n"))
    worst = -1
    best = -1
    best_value = -1
    worst_value = -1
    average_time = 0
    average_success = 0
    average_distances = []
    points = generate_points()
    for i in range(n):
        clusters= []
        start = time.time()
        #CENTROIDY A MEDOIDY
        if a == 1 or a == 2:
            k_means(clusters, points, k, a)
        # DIVIZIVNE
        if a == 3:
            divisive_clustering(clusters, points, k)
        # AGLOMERATIVNE
        if a == 4:
            agglomerative_clustering(clusters, points, k)
        end = time.time()
        average_time += end-start
        print("Time ", end - start)
        if draw_flag == 1:
            draw(clusters)
        if points_flag == 2 and a != 4:
            print("Idem cistit")
            clean_points(points)
        if points_flag == 1 or a == 4:
            points = generate_points()
        val = evaluate(clusters, average_distances)
        average_success += val
        print("Average distances ", average_distances[-1])
        print("Average distance of a cluster", sum(average_distances[-1]) / k)
        if best == -1 or best_value < val:
            best = clusters
            best_value = val
        elif worst == -1 or worst_value > val:
            worst = clusters
            worst_value = val
        print("="*30)
    if a==1 or a==2:
        print("Average number of iterations ", iterations_avg/n)
    print("Average time", average_time / n)
    print("Average success", average_success / n)
    print("Overall average distance", calculate_overall_average_distance(average_distances))
    print("Best  success ", best_value)
    print("Worst success ", worst_value)
    if draw_flag == 2:
        draw(worst)
        draw(best)

def calculate_overall_average_distance(average_distances):
    s = 0
    for distances in average_distances:
        s += sum(distances)/len(distances)
    return s / len(average_distances)


def main():
    global number_of_points
    global DEBUG
    de = input("Debug y/n")
    if de == "y":
        DEBUG = True
    else:
        DEBUG = False
    number_of_points = int(input("Number of points added to the first 20 points\n"))
    testing_function()


if __name__ == '__main__':
    main()


