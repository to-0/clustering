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
colors = ["red", "green", "blue", "orange", "purple", "brown", "black", "pink", "silver", "rosybrown",
          "firebrick", "darksalmon", "sienna", "gold", "palegreen", "deepskyblue", "navy", "mediumpurple",
          "plum", "palevioletred"]
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
    def __init__(self, x, y, cluster,i):
        self.x = x
        self.y = y
        self.cluster = cluster
        self.index = i

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

    # def medoid_is_the_new_centroid(self):
    #     x = self.x_sum / len(self.cluster_points)
    #     y = self.y_sum / len(self.cluster_points)
    #     min_distance = -1
    #     min_point = -1
    #     for point in self.cluster_points:
    #         dist = calculate_distance(point.x, point.y, x, y)
    #         if min_distance == -1 or min_distance > dist:
    #             min_distance = dist
    #             min_point = point
    #     return [min_point.x, min_point.y]
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


    # def calculate_medoid_experiment(self):
    #
    #     max_y = max(point.y for point in self.cluster_points)
    #     max_x = max(point.x for point in self.cluster_points)
    #
    #     min_y = min(point.y for point in self.cluster_points)
    #     min_x = min(point.x for point in self.cluster_points)
    #
    #
    #     med = [abs(max_x-min_x)/2, abs(max_y-min_y)/2]
    #
    #     min_dist = -1
    #     min_point = -1
    #     for point in self.cluster_points:
    #         dist = calculate_distance(point.x,point.y,med[0],med[0])
    #         if min_dist == -1 or dist < min_dist:
    #             min_dist = dist
    #             min_point = point
    #     return min_point


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

    # def update_matrix_add_point(self, point):
    #     self.cluster_points.append(point)
    #     distances = []
    #     for p in self.cluster_points:
    #         if p == point:
    #             continue
    #         dist = calculate_distance(point.x, point.y, p.x, p.y)
    #         distances.append(dist)
    #     self.matrix.append(distances)
    #     i = 0
    #     for row in self.matrix:
    #         if i == len(distances):
    #             row.append(0)
    #         else:
    #             row.append(distances[i])
    #         i += 1
    #     self.matrix[-1].append(0)
    #
    # def update_matrix_remove_point(self, point):
    #     point_index = self.cluster_points.index(point)
    #     # delete row
    #     self.matrix.remove(self.matrix[point_index])
    #     # delete column
    #     for i in range(len(self.matrix)):
    #         self.matrix[i].remove(self.matrix[i][point_index])
    #     self.cluster_points.remove(point)


def generate_points():
    points = []
    # prvych 20
    used_x = []
    used_y = []
    for i in range(20):
        x = random.randint(-5000, 5000)
        y = random.randint(-5000, 5000)
        points.append(Point(x, y, 0, i))

    #create_first_20()
    #global points
    print(len(points))
    for i in range(number_of_points):
        index = random.randint(0, len(points)-1)
        #print("dlzka je ", len(points), index)
        p = points[index]
        x_offset = random.randint(-100, 100)
        y_offset = random.randint(-100, 100)
        point = Point(p.x+x_offset, p.y+y_offset, 0,i+20)
        points.append(point)
    return points

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distance


# vyberiem nahodne k bodov ako k clustrov
def select_k_clusters(clusters, k, points):
    used = []
    counter = 0
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

def recalculate_distances(old_cluster, new_cluster, distance_list, clusters, point):
    index_new = clusters.index(new_cluster)
    if old_cluster != 0:
        index_old = clusters.index(old_cluster)
        i = 0
        for p2 in old_cluster.cluster_points:
            dist = calculate_distance(point.x, point.y, p2.x, p2.y)
            distance_list[index_old][i] -= dist
            i += 0
    i = 0
    for p2 in new_cluster.cluster_points:
        dist = calculate_distance(point.x, point.y, p2.x, p2.y)
        distance_list[index_new][i] += dist
        i += 0



def k_means(clusters, points, k, cent_med):
    # init step
    global time_centers
    global time_matrix_things
    select_k_clusters(clusters, k, points)
    print(clusters)
    iterations = 0
    global time_distances
    while True:
        print("Iteracia", iterations)
        changes = 0
        #STEP 1
        print("Idem ratat centroidy/medoidy")
        start_centeres = time.time()
        if cent_med == 1:
            cluster_centers = compute_centroids(clusters)
        else:
            cluster_centers = compute_all_medoids(clusters)
        #cluster_centers = compute_cluster_centers(cent_med, clusters)
        end_centers = time.time()
        time_centers += end_centers - start_centeres
        print("Doratal som")
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
            # CENTROID
            # if cent_med == 1:
            #     if old_cluster != 0:
            #         old_cluster.cluster_points.remove(point)
            #         old_cluster.x_sum -= point.x
            #         old_cluster.y_sum -= point.y
            #     min_cluster.cluster_points.append(point)
            #     min_cluster.x_sum += point.x
            #     min_cluster.y_sum += point.y
            #
            # # MEDOID
            # if cent_med == 2:
            #     if old_cluster != 0:
            #         old_cluster.cluster_points.remove(point)
            #     min_cluster.cluster_points.append(point)
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
    print("Iteracii ", iterations)

def medoid_pam(clusters):
    medoids = []
    for cluster in clusters:
        for point in cluster.cluster_points:
            pass


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
    # clusters.append(Cluster(points, 0))
    # vytvorim si random 2 clustre uz ich rozdelim ako keby
    k_means(clusters, points, 2, 1)
    length = len(clusters)
    while length != k:
        # najdem si najhorsi cluster ten budem delit
        worst = calculate_worst_cluster(clusters)
        # toto mi prida dalsie 2 clustre cize ten stary cluster najhorsi mozem zahodit
        k_means(clusters, worst.cluster_points, 2, 1)
        clusters.remove(worst)
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
    print("Idem vytvarat maticu")
    create_distance_matrix(points)
    print("Vytvoril som maticu")
    # pre kazdy bod spravim cluster
    for point in points:
        clusters.append(Cluster([point]))
    print("priradl som")
    length = len(clusters)
    iteration = 0
    while length != k:
        print("Iteracia ", iteration)
        # najdem najblizsiu dvojicu
        closest = find_closest_couple()
        print("Nasiel som najblizsie ", closest)
        if closest[0] == closest[1]:
            print("Zle daco naslo")
            exit(1)
        # necham si ten prvy index a ten druhy zahodim
        new_cluster = clusters[closest[0]]
        second_cluster = clusters[closest[1]]
        # prekopirujem body z toho jedneho clustra co idem vymazat do druheho ktory mi zostane
        print("Idem prekopirovat body a je ich ", len(second_cluster.cluster_points))
        j = 0
        for point in second_cluster.cluster_points:
            new_cluster.cluster_points.append(point)
            point.cluster = new_cluster
            print("K je ", j)
            new_cluster.x_sum += point.x
            new_cluster.y_sum += point.y
            j += 1
        # vymazem jeden z tych clustrov zo zoznamu
        print("Idem vymazat?")
        clusters.remove(second_cluster)
        del second_cluster
        # updatnem maticu vymazem 1 riadok a stlpec a prepocitam jeden riadok a stlpec
        print("Idem updatnut maticu")
        update_matrix_after_removal(closest[0], closest[1], new_cluster, clusters)
        iteration += 1
        length = len(clusters)
        print("Length ", length)


def evaluate(clusters):
    all_clusters = len(clusters)
    bad_clusters = 0
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
    good_clusters = all_clusters - bad_clusters
    print("Uspesnost ", good_clusters/all_clusters*100)


def main():
    global number_of_points
    number_of_points = int(input("Number of points added to the first 20 points\n"))
    points = generate_points()
    #create_distance_matrix(points)
    print("1. K means clustering with centroid")
    print("2 K means clustering with medoid")
    print("3. Divisive clustering with centroid")
    print("4. Aglomerative clustering with centroid")
    choice = int(input())
    clusters = []
    k = int(input("K \n"))
    start = time.time()
    if choice == 1:
        k_means(clusters, points, k, 1)
    elif choice == 2:
        start = time.time()
        k_means(clusters, points, k, 2)
    elif choice == 3:
        divisive_clustering(clusters, points, k)
    elif choice == 4:
        agglomerative_clustering(clusters,points,k)
    end = time.time()
    a = end - start
    print("Idem kreslit")
    print(f'Cas remove a append {time_add_remove}')
    print(f'Cas medoidu {time_medoid}')
    print(f'Celkovy cas {a}')
    print(f' Cas na ratanie medoidov/centroidov {time_centers}')
    print(f'Cas na ratanie vzdialenosti {time_distances}')
    print(f'Cas na handlovanie matice {time_matrix_things}')
    evaluate(clusters)
    draw(clusters)

if __name__ == '__main__':
    main()


