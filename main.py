import math
import random
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
import time

border = 10000
time_add_remove = 0
all_time = 0
time_medoid = 0
time_centers = 0
time_distances = 0
time_matrix_things = 0
colors = ["red", "green", "blue", "orange", "purple", "brown", "black", "pink","silver","rosybrown",
          "firebrick","darksalmon", "sienna","gold","palegreen","deepskyblue","navy","mediumpurple",
          "plum","palevioletred"]
distance_matrix = list()

def create_distance_matrix(points):
    global distance_matrix
    n = len(points)
    distance_matrix = np.zeros((n, n))
    i = 0
    for p1 in points:
        j = 0
        for p2 in points:
            dist = calculate_distance(p1.x, p1.y, p2.x, p2.y)
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

    def medoid_is_the_new_centroid(self):
        x = self.x_avg / len(self.cluster_points)
        y = self.y_avg / len(self.cluster_points)
        min_distance = -1
        min_point = -1
        for point in self.cluster_points:
            dist = calculate_distance(point.x,point.y,x,y)
            if min_distance == -1 or min_distance > dist:
                min_distance = dist
                min_point = point
        return [min_point.x,min_point.y]


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


    def calculate_medoid_experiment(self):

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

    def update_matrix_add_point(self, point):
        self.cluster_points.append(point)
        distances = []
        for p in self.cluster_points:
            if p == point:
                continue
            dist = calculate_distance(point.x, point.y, p.x, p.y)
            distances.append(dist)
        self.matrix.append(distances)
        i = 0
        for row in self.matrix:
            if i == len(distances):
                row.append(0)
            else:
                row.append(distances[i])
            i += 1
        self.matrix[-1].append(0)

    def update_matrix_remove_point(self, point):
        point_index = self.cluster_points.index(point)
        # delete row
        self.matrix.remove(self.matrix[point_index])
        # delete column
        for i in range(len(self.matrix)):
            self.matrix[i].remove(self.matrix[i][point_index])
        self.cluster_points.remove(point)



#points = []
def create_first_20(points):
    used_x = []
    used_y = []
    for i in range(20):
        x = random.randint(-5000, 5000)
        y = random.randint(-5000, 5000)
        points.append(Point(x, y, 0, i))


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
    for i in range(20000):
        index = random.randint(0, len(points)-1)
        #print("dlzka je ", len(points), index)
        p = points[index]
        x_offset = random.randint(-100, 100)
        y_offset = random.randint(-100, 100)
        point = Point(p.x+x_offset, p.y+y_offset, 0,i+20)
        points.append(point)
    return points

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))
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
        cluster = Cluster([points[c_number]], i)
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
            color = colors[i%len(colors)]
            plt.plot(point.x, point.y, marker=".", markersize=2, color=color)
        i+= 1

    plt.show()
    # for point in points:
    #     #print("Col ",point.cluster)
    #     color = colors[point.cluster.color]
    #     plt.plot(point.x, point.y, marker=".", markersize=2, color=color)
    #
    # plt.show()

def compute_cluster_centers(cent_med, clusters):
    centers = []
    for cluster in clusters:
        if cent_med == 1:
            centroid = cluster.calculate_centroid()
            centers.append(centroid)
        else:
            #medoid = cluster.calculate_medoid_test()
            medoid = cluster.calculate_medoid_matrix()
            centers.append([medoid.x, medoid.y])
    return centers


def k_means(clusters, points, k, cent_med):
    # init step
    global time_centers
    global time_matrix_things
    select_k_clusters(clusters, k,points)
    print(clusters)
    iterations = 0
    global time_distances
    while True:
        print("Iteracia", iterations)
        changes = 0
        #STEP 1
        print("Idem ratat centroidy/medoidy")
        start_centeres = time.time()
        cluster_centers = compute_cluster_centers(cent_med, clusters)
        print(cluster_centers)
        end_centers = time.time()
        time_centers += end_centers - start_centeres
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
                # s = time.time()
                # if old_cluster != 0:
                #     old_cluster.update_matrix_remove_point(point)
                # min_cluster.update_matrix_add_point(point)
                # e = time.time()
                # time_matrix_things += e-s
            point.cluster = min_cluster
        if changes == 0:
            break
        iterations += 1
    print("Iteracii ", iterations)



def divisive_clustering(clusters, k):
    # clusters.append(Cluster(points, 0))
    k_means(clusters,2, 1)
    length = len(clusters)
    while length != k:
        pass

def main():
    points = generate_points()
    #create_distance_matrix(points)
    print("1. K means clustering with centroid")
    print("2 K means clustering with medoid")
    print("3. Divisive clustering with centroid")
    print("4. Aglomerative clustering with centroid")
    choice = int(input())
    clusters = []
    k = int(input("K \n"))
    if choice == 1:
        start = time.time()
        k_means(clusters, points, k, 1)
        end = time.time()
    elif choice == 2:
        start = time.time()
        k_means(clusters, points, k, 2)
        end = time.time()
    elif choice == 3:
        divisive_clustering(clusters, k)
    a = end - start
    print("Idem kreslit")
    print(f'Cas remove a append {time_add_remove}')
    print(f'Cas medoidu {time_medoid}')
    print(f'Celkovy cas {a}')
    print(f' Cas na ratanie medoidov/centroidov {time_centers}')
    print(f'Cas na ratanie vzdialenosti {time_distances}')
    print(f'Cas na handlovanie matice {time_matrix_things}')

    draw(clusters)

if __name__ == '__main__':
    main()


