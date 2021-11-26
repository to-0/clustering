import math
import random
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
import time

border = 10000
clusters = []
time_add_remove = 0
all_time = 0
time_medoid = 0
colors = ["red", "green", "blue", "orange", "purple", "brown", "black", "pink","silver","rosybrown","firebrick","darksalmon", "sienna",
          "gold","palegreen","deepskyblue","navy","mediumpurple","plum","palevioletred"]

class Point:
    def __init__(self, x, y, cluster):
        self.x = x
        self.y = y
        self.cluster = cluster

class Cluster:
    def __init__(self, cluster_points, color):
        self.cluster_points = cluster_points
        self.color = color
        self.x_avg = 0
        self.y_avg = 0
        self.initialize_averages()
        # test
        self.min_x = 0
        self.min_y = 0
        self.max_x = 0
        self.max_y = 0
        #
        self.matrix = self.create_matrix()

    def initialize_averages(self):
        for point in self.cluster_points:
            self.x_avg += point.x
            self.y_avg += point.y


    def calculate_centroid(self):
        x = self.x_avg/len(self.cluster_points)
        y = self.y_avg/len(self.cluster_points)
        return [x, y]


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

    def calculate_medoid_experiment(self):
        # sorted_y = sorted(point.y for point in self.cluster_points)
        # sorted_x = sorted(point.x for point in self.cluster_points)
        # max_x = sorted_x[-1]
        # max_y = sorted_y[-1]
        # min_x = sorted_x[0]
        # min_y = sorted_y[0]

        max_y = max(point.y for point in self.cluster_points)
        max_x = max(point.x for point in self.cluster_points)

        min_y = min(point.y for point in self.cluster_points)
        min_x = min(point.x for point in self.cluster_points)


        med = [abs(max_x-min_x)/2, abs(max_y-min_y)/2]

        min_dist = -1
        min_point = -1
        for point in self.cluster_points:
            dist = calculate_distance(point.x,point.y,med[0],med[0])
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                min_point = point
        return min_point


    def update_centroid_remove_point(self, point):
        self.x_avg -= point.x
        self.y_avg -= point.y
        self.cluster_points.remove(point)
        self.centroid = [self.x_avg/len(self.cluster_points), self.y_avg/len(self.cluster_points)]


    def calculate_medoid(self):
        # vrati cislo najmensieho riadku
        index_medoid = np.argmin(self.matrix.sum(0))
        print(self.matrix)
        return self.cluster_points[index_medoid]

    def create_matrix(self):
        length = len(self.cluster_points)
        matrix = [[0 for x in range(length)] for y in range(length)]
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
    def update_matrix_add_point_test(self,point):
        old_length = len(self.cluster_points)
        self.cluster_points.append(point)
        distances = []
        for p in self.cluster_points:
            # seba preskoci to bude nula aj tak
            if p == point:
                continue
            dist = calculate_distance(point.x, point.y, p.x, p.y)
            distances.append(dist)

    def update_matrix_add_point(self, point):
        i = 0
        old_length = len(self.cluster_points)
        self.cluster_points.append(point)

        if old_length == 1:
            self.matrix = np.zeros((2, 2))
            p1 = self.cluster_points[0]
            p2 = self.cluster_points[1]
            dist = calculate_distance(p1.x,p1.y,p2.x,p2.y)
            self.matrix[0][1] = dist
            self.matrix[1][0] = dist
        else:
            new_length = len(self.cluster_points)
            distances = []
            for p in self.cluster_points:
                # seba preskoci to bude nula aj tak
                if p == point:
                    continue
                dist = calculate_distance(point.x, point.y, p.x, p.y)
                distances.append(dist)
                i += 1
            print(distances)
            print(self.matrix)
            self.matrix = np.append(self.matrix, np.array([distances]), 0)
            self.matrix = np.append(self.matrix, np.array([distances]), 1)

    def update_matrix_remove_point(self, point):
        point_index = self.cluster_points.index(point)
        # delete row
        self.matrix = np.delete(self.matrix, point_index, 0)
        # delete column
        self.matrix = np.delete(self.matrix, point_index, 1)
        self.cluster_points.remove(point)


points = []
def create_first_20():
    used_x = []
    used_y = []
    for i in range(20):
        x = random.randint(-5000, 5000)
        y = random.randint(-5000, 5000)
        points.append(Point(x, y, 0))


def generate_points():
    create_first_20()
    global points
    print(len(points))
    for i in range(20000):
        index = random.randint(0, len(points)-1)
        #print("dlzka je ", len(points), index)
        p = points[index]
        x_offset = random.randint(-100, 100)
        y_offset = random.randint(-100, 100)
        point = Point(p.x+x_offset, p.y+y_offset, 0)
        points.append(point)

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))
    return distance


def calculate_medoid(cluster):
    pass

def add_point_kmeans(point, cent_med):
    min_d = -1
    min_cluster = clusters[0]
    old_cluster = point.cluster
    global time_medoid
    global time_add_remove
    for cluster in clusters:
        if cent_med == 1:
            centroid = cluster.centroid
            dist = calculate_distance(point.x, point.y, centroid[0], centroid[1])
        else:
            start = time.time()
            medoid = cluster.calculate_medoid()
            end = time.time()
            time_medoid += end-start
            dist = calculate_distance(point.x, point.y, medoid.x, medoid.y)
        #print("Distance ", dist)
        if min_d == -1:
            min_d = dist
            min_cluster = cluster
        elif dist < min_d:
            min_d = dist
            min_cluster = cluster

    if min_cluster == point.cluster:
        return 0

    point.cluster = min_cluster
    # centroid
    if cent_med == 1:
        if old_cluster != 0:
            old_cluster.update_centroid_remove_point(point)
        min_cluster.cluster_points.append(point)
        min_cluster.update_centroid()
    # medoid
    elif cent_med == 2:
        start = time.time()
        # if old_cluster != 0:
        #     old_cluster.cluster_points.remove(point)
        # min_cluster.cluster_points.append(point)
        if old_cluster != 0:
            old_cluster.update_matrix_remove_point(point)
        min_cluster.update_matrix_add_point(point)
        end = time.time()
        time_add_remove += end - start
    return 1

# vyberiem nahodne k bodov ako k clustrov
def select_k_clusters(k):
    global clusters
    used = []
    counter = 0
    for i in range(k):
        c_number = random.randint(0, len(points)-1)
        while c_number in used:
            c_number = random.randint(0, len(points)-1)
        used.append(c_number)
        cluster = Cluster([points[c_number]], i)
        clusters.append(cluster)
        points[c_number].cluster = cluster



def draw():
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.yticks(np.arange(-5000, 6000, 1000))
    plt.yticks(np.arange(-5000, 6000, 1000))
    #fig, axs = plt.subplots()  # a figure with a 2x2 grid of Axes
    #print(axs)
    for point in points:
        #print("Col ",point.cluster)
        color = colors[point.cluster.color]
        plt.plot(point.x, point.y, marker=".", markersize=2, color=color)

    plt.show()

def compute_cluster_centers(cent_med):
    centers = []
    for cluster in clusters:
        if cent_med == 1:
            centroid = cluster.calculate_centroid()
            centers.append(centroid)
        else:
            medoid = cluster.calculate_medoid_test()
            centers.append([medoid.x, medoid.y])
    return centers


def k_means(k, cent_med):
    # init step
    select_k_clusters(k)
    print(clusters)
    iterations = 0
    while True:
        print("Iteracia",iterations)
        changes = 0
        #STEP 1
        print("Idem ratat centroidy/medoidy")
        cluster_centers = compute_cluster_centers(cent_med)
        print("Doratal som")
        # for cluster in clusters:
        #     for point in cluster.cluster_points:
        #         print(point.x, point.y)
        # for center in cluster_centers:
        #      print(center)

        for point in points:
            old_cluster = point.cluster
            min_dist = -1
            min_cluster = -1
            for i in range(len(cluster_centers)):
                center = cluster_centers[i]
                dist = calculate_distance(point.x, point.y, center[0], center[1])
                #print("dist", dist)
                if min_cluster == -1 or dist < min_dist:
                    min_dist = dist
                    min_cluster = clusters[i]

            if old_cluster != min_cluster:
                changes += 1
            # CENTROID
            if cent_med == 1:
                if old_cluster != 0:
                    old_cluster.cluster_points.remove(point)
                    old_cluster.x_avg -= point.x
                    old_cluster.y_avg -= point.y
                min_cluster.cluster_points.append(point)
                min_cluster.x_avg += point.x
                min_cluster.y_avg += point.y
            # MEDOID
            if cent_med == 2:
                if old_cluster != 0:
                    old_cluster.cluster_points.remove(point)
                min_cluster.cluster_points.append(point)
                # if old_cluster != 0:
                #     old_cluster.update_matrix_remove_point(point)
                # min_cluster.update_matrix_add_point(point)
            point.cluster = min_cluster
        if changes == 0:
            break
        iterations += 1
    print("Iteracii ", iterations)


def k_means_clustering(k, cent_med):

    print(clusters)
    # for cluster in clusters:
    #     p = cluster.cluster_points[0]
    #     print(f'{p.x},{p.y}')
    #     print(cluster.centroid)
    # zvysnych N-k priradim ku clustrom
    while True:
        changes = 0
        for point in points:
            changes += add_point_kmeans(point, cent_med)
        if changes == 0:
            break
    print("Skoncili sme")
    for cluster in clusters:
        print(len(cluster.cluster_points))




def divisive_clustering(k):
    # clusters.append(Cluster(points, 0))
    k_means_clustering(2, 1)
    length = len(clusters)
    while length != k:
        pass


if __name__ == '__main__':
    generate_points()
    print("1. K means clustering with centroid")
    print("2 K means clustering with medoid")
    print("3. Divisive clustering with centroid")
    print("4. Aglomerative clustering with centroid")
    choice = int(input())
    if choice == 1:
        k = int(input("K \n"))
        start = time.time()
        k_means(8, 1)
        end = time.time()
    elif choice == 2:
        k = int(input("K \n"))
        start = time.time()
        k_means(8, 2)
        end = time.time()
    elif choice == 3:
        divisive_clustering()
    a = end-start
    print("Idem kreslit")
    print(f'Cas remove a append {time_add_remove}')
    print(f'Cas medoidu {time_medoid}')
    print(f'Celkovy cas {a}')

    draw()


